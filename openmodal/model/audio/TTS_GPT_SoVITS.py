import re
from typing import Dict
import torch
import soundfile
import numpy as np
from tqdm import tqdm
from pytorch_lightning import LightningModule

from openmodal.component.audio.text2Semantic_GPT_SoVITS import Text2SemanticDecoder
from openmodal.engine import ModelBase
from openmodal.model import BaseModel
from openmodal.process.text.text_cleaner import clean_text, cleaned_text_to_sequence
from openmodal.util.text import split_sentence
from openmodal.util.text.languages.symbols import symbols as openmodal_symbols, \
    num_languages as openmodal_num_languages, language_tone_num_map
from openmodal.view_object.text.languages import LanguagesEnum


@ModelBase.register_module(name="GPTSoVITS_TTS")
class GPTSoVITS_TTS(BaseModel, LightningModule):
    """
    The GPTSoVITS_TTS model is open-sourced by
    https://github.com/RVC-Boss/GPT-SoVITS
    """

    def __init__(self,
                 language,
                 ckpt_path=None,
                 ckpt_bert_dir=None,
                 water_mark=None,
                 is_train=False,
                 is_half =True,
                 *args,
                 **kwargs):
        super().__init__(device=None, *args, **kwargs)
        if ckpt_path is not None:
            # load state_dict
            checkpoint_dict, hps = self.load_or_download_model(ckpt_path, self.device)

            num_languages = hps.get('num_languages', openmodal_num_languages)
            num_tones = hps.get('num_tones', language_tone_num_map[language])
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

        model = Text2SemanticDecoder(
            model_dim=model_dim,
            embedding_dim=embedding_dim,
            num_head=num_head,
            num_layers=num_layers,
            vocab_size=vocab_size,
            phoneme_vocab_size=phoneme_vocab_size,
            p_dropout=p_dropout,
            EOS=EOS,
        )
        self.model = model
        self.hps = hps
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}

        if is_train:
            self.automatic_optimization = False
            self.save_hyperparameters()
            model.train()
            if pretrained_s1:
                print(self.load_state_dict(torch.load(pretrained_s1, map_location="cpu")["weight"]))
        else:
            self.load_state_dict(checkpoint_dict, strict=True)
            model.eval()

        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language  # we support a ZH_MIX_EN model
        self.ckpt_bert_dir = ckpt_bert_dir

        if is_half:
            self.half()
        self.to(self.device)
        total = sum([param.nelement() for param in self.parameters()])
        print("\nName of model: %s" % (self._get_name()))
        print("Number of parameter: %.2fM" % (total / 1e6))

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

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    def forward(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8,
                speed=1.0, format=None):
        language = self.language

        texts = split_sentence(text, language=language)
        print(" > Text split to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")

        audio_list = []
        for t in tqdm(texts):
            if language in [LanguagesEnum.EN, LanguagesEnum.ZH_MIX_EN]:
                # split the audio by capital letters
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(t, language, self.hps, self.ckpt_bert_dir,
                                                                            device,
                                                                            self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=1. / speed,
                )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                #
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is not None:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
        return audio


def get_text_for_tts_infer(text, language_str, hps, ckpt_bert_dir, device, symbol_to_id=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str, ckpt_bert_dir)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    if hps.data.add_blank:
        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language = intersperse(language, 0)
        # todo: 是否需要针对phone和tone分别进行embedding，仍待考究
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    if getattr(hps.data, "disable_bert", False):
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    else:
        bert = get_bert(norm_text, word2ph, language_str, ckpt_bert_dir, device)
        del word2ph
        assert bert.shape[-1] == len(phone), phone

        if language_str == "ZH":
            bert = bert
            ja_bert = torch.zeros(768, len(phone))
        elif language_str in ["JP", "EN", "ZH_MIX_EN", 'KR', 'SP', 'ES', 'FR', 'DE', 'RU']:
            ja_bert = bert
            bert = torch.zeros(1024, len(phone))
        else:
            raise NotImplementedError()

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def get_bert(norm_text, word2ph, language, ckpt_bert_dir, device):
    from openmodal.model.text.pretrained_bert.chinese_bert import get_bert_feature as zh_bert
    from openmodal.model.text.pretrained_bert.english_bert import get_bert_feature as en_bert
    from openmodal.model.text.pretrained_bert.japanese_bert import get_bert_feature as jp_bert
    from openmodal.util.text.languages.chinese_mix import get_bert_feature as zh_mix_en_bert
    from openmodal.model.text.pretrained_bert.spanish_bert import get_bert_feature as sp_bert
    from openmodal.model.text.pretrained_bert.french_bert import get_bert_feature as fr_bert
    from openmodal.util.text.languages.korean import get_bert_feature as kr_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert, 'ZH_MIX_EN': zh_mix_en_bert,
                          'FR': fr_bert, 'SP': sp_bert, 'ES': sp_bert, "KR": kr_bert}
    bert = lang_bert_func_map[language](norm_text, word2ph, ckpt_bert_dir, device)
    return bert
