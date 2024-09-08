import torch
import numpy as np
import soundfile
import os
import librosa

from openmodal.component.audio.GPTSoVITS import GPTSoVITS
from openmodal.component.audio.cnhubert import CNHubert
from openmodal.engine import ModelBase
from openmodal.model import BaseModel
from openmodal.model.audio import GPTSoVITS_TTS
from openmodal.process.audio import speech_embedding
from openmodal.util.text import split_sentence


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
            f"{ckpt_path}/s2G2333k.pth", self.device)
        self.hps = hps

        self.tts_model = GPTSoVITS_TTS(
            language=language,
            ckpt_path=ckpt_path,
            ckpt_bert_path=ckpt_bert_path,
            is_train=is_train,
            is_half=self.is_half
        )
        self.ssl_model = CNHubert(base_path=ckpt_hubert_path)
        self.vq_model = GPTSoVITS(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )

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

    def extract_se(self, ref_wav_path, sr=16000, min_sec=3, max_sec=30, se_save_path=None):
        if isinstance(ref_wav_path, str):
            ref_wav_list = [ref_wav_path]

        gs = []
        for fname in ref_wav_list:
            with torch.no_grad() if not self.is_train else torch.enable_grad():
                # load reference wav
                ref_wav, sr = librosa.load(fname, sr=sr)
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
                gs.append(prompt.detach())

        gs = torch.stack(gs).mean(0)

        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(gs.cpu(), se_save_path)

        gs = gs.to(self.device)
        return gs

    def forward(self,
                text,
                references_path,
                reference_text,
                speaker,
                output_path=None,
                tau=0.3,
                source_sr=None,
                *args,
                **kwargs):
        hps = self.hps

        # process the reference audio and text
        prompt = self.extract_se(references_path)
        phones_ref, bert_ref, norm_text_ref = get_text_for_tts_infer(t, language, self.hps, self.ckpt_bert_path,
                                                                            device,
                                                                            self.symbol_to_id)

        texts = split_sentence(text, language=self.language)
        print(" > Text split to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            speaker_key = speaker.lower().replace('_', '-')
            src_se = torch.load(f"{self.ckpt_converter_path}/base_speakers/ses/{speaker_key}.pth",
                                map_location=self.device)
            audio = self.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=target_se, tau=tau)[0][
                0, 0].data.cpu().float().numpy()
            if self.watermark is not None:
                audio = self.add_watermark(audio, self.watermark)
            if output_path is None:
                return audio
            else:
                soundfile.write(output_path, audio, hps.data.sampling_rate)

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


hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.1:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.1:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.real.pow(2).sum(-1) + 1e-6)
    return spec


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
