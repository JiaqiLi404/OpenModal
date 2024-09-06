import re
import os
import json
import torch
import soundfile
import numpy as np
from tqdm import tqdm

from huggingface_hub import hf_hub_download

from openmodal.engine import ModelBase
from openmodal.model import BaseModel
from openmodal.component.audio.SoVITS_openvoice import OpenVoiceSoVITS
from openmodal.process.text.text_cleaner import clean_text, cleaned_text_to_sequence
from openmodal.util.text import split_sentence
from openmodal.view_object.text.languages import LanguagesEnum


@ModelBase.register_module(name="MeloTTS")
class MeloTTS(BaseModel):
    """
    The MeloTTS model is open-sourced by the OpenVoice team.
    https://github.com/myshell-ai/OpenVoice
    """

    def __init__(self,
                 language,
                 ckpt_bert_dir=None,
                 ckpt_path=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # config_path =
        hps = load_or_download_config(ckpt_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = OpenVoiceSoVITS(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(self.device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps

        # load state_dict
        checkpoint_dict = load_or_download_model(ckpt_path, self.device)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)

        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language  # we support a ZH_MIX_EN model
        self.ckpt_bert_dir = ckpt_bert_dir

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


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def load_or_download_config(config_path):
    if os.path.exists(os.path.join(config_path, "config.json")):
        config_path = os.path.join(config_path, "config.json")
    else:
        config_path = hf_hub_download(repo_id=config_path, filename="config.json")
    return get_hparams_from_file(config_path)


def load_or_download_model(ckpt_path, device):
    if os.path.exists(os.path.join(ckpt_path, "checkpoint.pth")):
        ckpt_path = os.path.join(ckpt_path, "checkpoint.pth")
    else:
        ckpt_path = hf_hub_download(repo_id=ckpt_path, filename="checkpoint.pth")
    return torch.load(ckpt_path, map_location=device)


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
