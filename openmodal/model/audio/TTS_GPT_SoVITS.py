import re
from typing import Dict

import torch
from pytorch_lightning import LightningModule

from openmodal.block.lr_schedulers import WarmupCosineLRSchedule
from openmodal.block.optim import ScaledAdam
from openmodal.component.audio.text2Semantic_GPT_SoVITS import Text2SemanticDecoder
from openmodal.engine import ModelBase
from openmodal.model import BaseModel
from openmodal.util.text.languages.symbols import symbols as openmodal_symbols, \
    num_languages as openmodal_num_languages, language_tone_num_map
from openmodal.view_object.text.languages import LanguagesEnum


@ModelBase.register_module(name="GPTSoVITS_TTS")
class GPTSoVITS_TTS(BaseModel, LightningModule):
    """
    The GPTSoVITS_TTS Text2Semantic model is open-sourced by
    https://github.com/RVC-Boss/GPT-SoVITS
    """

    def __init__(self,
                 language,
                 ckpt_path=None,
                 ckpt_bert_path=None,
                 ckpt_hubert_path=None,
                 water_mark=None,
                 is_train=False,
                 is_half=True,
                 symbols=None,
                 *args,
                 **kwargs):
        super().__init__(device=None, *args, **kwargs)
        if ckpt_path is not None:
            # load state_dict
            checkpoint_dict, hps = self.load_or_download_model(
                f"{ckpt_path}/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", self.device)
            self.hps = hps

            num_languages = hps.get('num_languages', openmodal_num_languages)
            num_tones = hps.get('num_tones', language_tone_num_map[language])
            if symbols is None:
                symbols = hps.get('symbols', openmodal_symbols)

            model_dim = hps.model.get("hidden_dim", None)
            embedding_dim = hps.model.get("embedding_dim", None)
            num_head = hps.model.get("head", None)
            num_layers = hps.model.get("n_layer", None)
            vocab_size = hps.model.get("vocab_size", len(symbols))
            phoneme_vocab_size = hps.model.get("phoneme_vocab_size", None)
            p_dropout = hps.model.get("dropout", None)
            EOS = hps.model.get("EOS", None)

            pretrained_s1 = hps.get("pretrained_s1", None)
        else:
            self.hps = None

            num_languages = openmodal_num_languages
            num_tones = language_tone_num_map[language]
            symbols = openmodal_symbols
            model_dim = 512
            embedding_dim = 512
            num_head = 16
            num_layers = 24
            vocab_size = len(symbols)
            phoneme_vocab_size = 732
            p_dropout = 0
            EOS = 0

        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}

        self.model = Text2SemanticDecoder(
            model_dim=model_dim,
            embedding_dim=embedding_dim,
            num_head=num_head,
            num_layers=num_layers,
            vocab_size=vocab_size,
            phoneme_vocab_size=phoneme_vocab_size,
            p_dropout=p_dropout,
            EOS=EOS,
        )

        if is_train:
            self.automatic_optimization = False
            self.save_hyperparameters()
            self.model.train()
            if pretrained_s1:
                print(self.load_state_dict(torch.load(pretrained_s1, map_location="cpu")["weight"]))
        else:
            self.load_state_dict(checkpoint_dict, strict=True)
            self.model.eval()

        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language  # we support a ZH_MIX_EN model
        self.ckpt_bert_path = ckpt_bert_path

        self.to(self.device) if self.device else None
        total = sum([param.nelement() for param in self.parameters()])
        print("\nName of model: %s" % (self._get_name()))
        print("Number of parameter: %.2fM" % (total / 1e6))

        self.ref_wav = None

    def training_step(self, batch: Dict, batch_idx: int):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        forward = self.model.forward if self.config["train"].get("if_dpo",
                                                                 False) == True else self.model.forward_old
        loss, acc = forward(
            batch["phoneme_ids"],
            batch["phoneme_ids_len"],
            batch["semantic_ids"],
            batch["semantic_ids_len"],
            batch["bert_feature"],
        )
        self.manual_backward(loss)
        if batch_idx > 0 and batch_idx % 4 == 0:
            opt.step()
            opt.zero_grad()
            scheduler.step()

        self.log(
            "total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            scheduler.get_last_lr()[0],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"top_{self.top_k}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def validation_step(self, batch: Dict, batch_idx: int):
        return

    def configure_optimizers(self):
        model_parameters = self.model.parameters()
        parameters_names = []
        parameters_names.append(
            [name_param_pair[0] for name_param_pair in self.model.named_parameters()]
        )
        lm_opt = ScaledAdam(
            model_parameters,
            lr=0.01,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )

        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": WarmupCosineLRSchedule(
                    lm_opt,
                    init_lr=self.config["optimizer"]["lr_init"],
                    peak_lr=self.config["optimizer"]["lr"],
                    end_lr=self.config["optimizer"]["lr_end"],
                    warmup_steps=self.config["optimizer"]["warmup_steps"],
                    total_steps=self.config["optimizer"]["decay_steps"],
                )
            },
        }

    def forward(self,
                all_phoneme_ids,
                all_phoneme_len,
                prompts,
                bert_feature,
                top_k: int = 20,
                top_p: float =0.6,
                early_stop_num: int = -1,
                temperature: float = 1.0,
                repetition_penalty: float = 1.35,
                **kwargs):
        return self.model.infer_panel_naive(all_phoneme_ids, all_phoneme_len, prompts, bert_feature, top_k,
                                     top_p, early_stop_num, temperature, repetition_penalty)
