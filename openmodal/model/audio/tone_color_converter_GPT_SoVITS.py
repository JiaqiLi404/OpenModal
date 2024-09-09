import torch
import numpy as np
import soundfile
import os
import librosa
import traceback
import LangSegment

from openmodal.component.audio.GPTSoVITS import GPTSoVITS
from openmodal.component.audio.cnhubert import CNHubert
from openmodal.engine import ModelBase
from openmodal.model import BaseModel
from openmodal.model.audio import GPTSoVITS_TTS
from openmodal.model.text.pretrained_bert import get_bert
from openmodal.process.audio import speech_embedding
from openmodal.process.text.text_cleaner import clean_text, cleaned_text_to_sequence
from openmodal.util.audio import load_audio
from openmodal.util.text import split_sentence
from openmodal.util.text.languages.symbols import symbols as openmodal_symbols
from openmodal.util.torch import spectrogram_torch
from openmodal.view_object.text.languages import LanguagesEnum


@ModelBase.register_module(name="SoVITSToneColorConverter")
class SoVITSToneColorConverter(BaseModel):
    def __init__(self,
                 language,
                 ckpt_path=None,
                 ckpt_bert_path=None,
                 ckpt_hubert_path=None,
                 ckpt_whisper_path=None,
                 water_mark=None,
                 is_train=False,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.ckpt_path = ckpt_path
        self.ckpt_bert_path = ckpt_bert_path
        self.ckpt_hubert_path = ckpt_hubert_path
        self.ckpt_whisper_path = ckpt_whisper_path
        self.language = language
        self.is_train = is_train

        # load state_dict
        checkpoint_dict, hps = self.load_or_download_model(
            f"{ckpt_path}/s2G2333k_symbols.pth", self.device)
        hps.model.semantic_frame_rate = "25hz"

        self.hps = hps
        self.symbols = hps.get("symbols", openmodal_symbols)
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}

        self.tts_model = GPTSoVITS_TTS(
            language=language,
            ckpt_path=ckpt_path,
            ckpt_bert_path=ckpt_bert_path,
            is_train=is_train,
            is_half=self.is_half,
            symbols=self.symbols,
        )
        self.ssl_model = CNHubert(base_path=ckpt_hubert_path)

        self.vq_model = GPTSoVITS(
            len(self.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        self.vq_model.load_state_dict(checkpoint_dict)

        self.watermark = water_mark

        if water_mark is not None:
            import wavmark
            self.watermark_model = wavmark.load_model()
        else:
            self.watermark_model = None

        if is_train:
            self.train()
        else:
            self.eval()
        self.to(self.device) if self.device else None
        self.half() if self.is_half else None

        self.cache = {}

    def extract_se(self, ref_wav_path, sr=16000, min_sec=3, max_sec=30, se_save_path=None):
        with torch.no_grad() if not self.is_train else torch.enable_grad():
            # load reference wav
            ref_wav, sr = librosa.load(ref_wav_path, sr=sr)
            if (ref_wav.shape[0] > max_sec * sr or ref_wav.shape[0] < min_sec * sr):
                raise ValueError(f"Reference wav length should be between {min_sec} and {max_sec} seconds.")
            ref_wav = torch.from_numpy(ref_wav)
            # add zero wave
            zero_wav = np.zeros(
                int(self.hps.data.sampling_rate * 0.3),
                dtype=np.float16 if self.is_half else np.float32,
            )
            zero_wav_torch = torch.from_numpy(zero_wav)
            if self.is_half:
                ref_wav = ref_wav.half()
                zero_wav_torch = zero_wav_torch.half()
            ref_wav = ref_wav.to(self.device)
            zero_wav_torch = zero_wav_torch.to(self.device)
            ref_wav = torch.cat([ref_wav, zero_wav_torch])
            ssl_content = self.ssl_model.model(ref_wav.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0)

        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(prompt.cpu(), se_save_path)

        return prompt

    def forward(self,
                text,
                references_path,
                reference_text,
                reference_language,
                speaker,
                output_path=None,
                tau=0.3,
                source_sr=None,
                use_cache=True,
                speed=1,
                top_k=15,
                top_p=1,
                temperature=1,
                *args,
                **kwargs):
        hps = self.hps
        dtype = torch.float16 if self.is_half == True else torch.float32

        with torch.no_grad() if not self.is_train else torch.enable_grad():
            # process the input text
            texts = split_sentence(text, language=self.language)
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")

            # process the reference audio and text
            if isinstance(references_path, str):
                references_path = [references_path]
            prompt = self.extract_se(references_path[0])

            if reference_text is not None:
                norm_text_ref, phone_ref, tone_ref, word2ph_ref = clean_text(reference_text, reference_language,
                                                                             self.ckpt_bert_path)
                phone_ref, tone_ref, language_ref = cleaned_text_to_sequence(phone_ref, tone_ref, reference_language,
                                                                             self.symbol_to_id, with_tone=True)
                bert_ref = get_bert(norm_text_ref, word2ph_ref, reference_language, self.ckpt_bert_path, self.device).to(
                    dtype)
                phone_ref = torch.LongTensor(phone_ref)
                tone_ref = torch.LongTensor(tone_ref)
                language_ref = torch.LongTensor(language_ref)

            audio_opt = []
            for i_text, text in enumerate(texts):
                bert_text, phone_text, norm_text_text = self.automatic_get_bert(text)
                # norm_text_text, phone_text, tone_text, word2ph_text = clean_text(text, self.language, self.ckpt_bert_path)
                # phone_text, tone_text, language_text = cleaned_text_to_sequence(phone_text, tone_text, self.language, self.symbol_to_id,with_tone=True)
                # bert_text = get_bert(norm_text_text, word2ph_text, self.language, self.ckpt_bert_path, self.device).to(dtype)
                # phone_text = torch.LongTensor(phone_text)
                # tone_text = torch.LongTensor(tone_text)
                # language_text = torch.LongTensor(language_text)
                all_phoneme_ids = torch.LongTensor(phone_text).to(self.device).unsqueeze(0)

                if reference_text is not None:
                    # combine the reference audio and text
                    bert = torch.cat([bert_ref, bert_text], dim=1)
                    all_phoneme_ids = torch.cat([torch.LongTensor(phone_ref).to(self.device).unsqueeze(0), all_phoneme_ids],
                                                dim=1)

                bert = bert.to(dtype).to(self.device).unsqueeze(0)
                all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

                if (i_text in self.cache and use_cache):
                    pred_semantic = self.cache[i_text]
                else:
                    pred_semantic, idx = self.tts_model.forward(
                        all_phoneme_ids,
                        all_phoneme_len,
                        prompt if reference_text is not None else None,
                        bert,
                        # prompt_phone_len=ph_offset,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=50 * self.tts_model.hps.data.max_sec,
                    )
                    pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                    self.cache[i_text] = pred_semantic if use_cache else None

                refers = []
                for filename in references_path:
                    try:
                        audio = load_audio(filename, int(hps.data.sampling_rate))
                        audio = torch.FloatTensor(audio)
                        maxx = audio.abs().max()
                        if (maxx > 1): audio /= min(2, maxx)
                        audio_norm = audio
                        audio_norm = audio_norm.unsqueeze(0)
                        refer = spectrogram_torch(
                            audio_norm,
                            hps.data.filter_length,
                            hps.data.sampling_rate,
                            hps.data.hop_length,
                            hps.data.win_length,
                            center=False,
                        )
                        refer = refer.to(dtype).to(self.device)
                        refers.append(refer)
                    except:
                        traceback.print_exc()
                audio = (
                self.vq_model.decode(pred_semantic, torch.LongTensor(phone_text).to(self.device).unsqueeze(0), refers,
                                     speed=speed).detach().cpu().numpy()[0, 0])
                max_audio = np.abs(audio).max()  # 简单防止16bit爆音
                if max_audio > 1: audio /= max_audio
                audio_opt.append(audio)
                zero_wav = np.zeros(
                    int(self.hps.data.sampling_rate * 0.3),
                    dtype=np.float16 if self.is_half else np.float32,
                )
                audio_opt.append(zero_wav)

            audio = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

            if self.watermark is not None:
                audio = self.add_watermark(audio, self.watermark)
            if output_path is None:
                return audio
            else:
                soundfile.write(output_path, audio, hps.data.sampling_rate)
            return audio

    def add_watermark(self, audio, message):
        if self.watermark_model is None:
            return audio
        device = self.device
        bits = string_to_bits(message).reshape(-1)
        n_repeat = len(bits) // 32

        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K: (coeff * n + 1) * K]
            if len(trunck) != K:
                print('Audio too short, fail to add watermark')
                break
            message_npy = bits[n * 32: (n + 1) * 32]

            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(device)[None]
                message_tensor = torch.FloatTensor(message_npy).to(device)[None]
                signal_wmd_tensor = self.watermark_model.encode(signal, message_tensor)
                signal_wmd_npy = signal_wmd_tensor.detach().cpu().squeeze()
            audio[(coeff * n) * K: (coeff * n + 1) * K] = signal_wmd_npy
        return audio

    def detect_watermark(self, audio, n_repeat):
        bits = []
        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K: (coeff * n + 1) * K]
            if len(trunck) != K:
                print('Audio too short, fail to detect watermark')
                return 'Fail'
            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(self.device).unsqueeze(0)
                message_decoded_npy = (
                        self.watermark_model.decode(signal) >= 0.5).int().detach().cpu().numpy().squeeze()
            bits.append(message_decoded_npy)
        bits = np.stack(bits).reshape(-1, 8)
        message = bits_to_string(bits)
        return message

    def automatic_get_bert(self, text):
        dtype = torch.float16 if self.is_half == True else torch.float32
        textlist = []
        langlist = []
        LangSegment.setfilters(["zh", "ja", "en", "ko"])
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = LanguagesEnum.from_str(langlist[i])
            norm_text, phone, tone, word2ph = clean_text(textlist[i], lang, self.ckpt_bert_path, is_g2pw=True,
                                                         is_g2pen=False)
            # with_tone=lang==LanguagesEnum.ZH or lang==LanguagesEnum.ZH_MIX_EN or lang==LanguagesEnum.ZH_CA
            phone, tone, language = cleaned_text_to_sequence(phone, tone, lang, self.symbol_to_id, with_tone=True)
            bert = get_bert(norm_text, word2ph, lang, self.ckpt_bert_path, self.device).to(dtype)
            # phone = torch.LongTensor(phone)
            # tone = torch.LongTensor(tone)
            # language = torch.LongTensor(language)
            # all_phoneme_ids = torch.LongTensor(phone).to(self.device).unsqueeze(0)

            phones_list.append(phone)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)
        return bert, phones, norm_text


def string_to_bits(string, pad_len=8):
    # Convert each character to its ASCII value
    ascii_values = [ord(char) for char in string]

    # Convert ASCII values to binary representation
    binary_values = [bin(value)[2:].zfill(8) for value in ascii_values]

    # Convert binary strings to integer arrays
    bit_arrays = [[int(bit) for bit in binary] for binary in binary_values]

    # Convert list of arrays to NumPy array
    numpy_array = np.array(bit_arrays)
    numpy_array_full = np.zeros((pad_len, 8), dtype=numpy_array.dtype)
    numpy_array_full[:, 2] = 1
    max_len = min(pad_len, len(numpy_array))
    numpy_array_full[:max_len] = numpy_array[:max_len]
    return numpy_array_full


def bits_to_string(bits_array):
    # Convert each row of the array to a binary string
    binary_values = [''.join(str(bit) for bit in row) for row in bits_array]

    # Convert binary strings to ASCII values
    ascii_values = [int(binary, 2) for binary in binary_values]

    # Convert ASCII values to characters
    output_string = ''.join(chr(value) for value in ascii_values)

    return output_string
