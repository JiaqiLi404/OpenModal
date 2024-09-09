import torch
import numpy as np
import soundfile
import os
import librosa

from openmodal.engine import ModelBase
from openmodal.component.audio.OpenVoiceSoVITS import OpenVoiceSoVITS
from openmodal.model import BaseModel
from openmodal.process.audio import speech_embedding
from openmodal.util.torch import spectrogram_torch


@ModelBase.register_module(name="OpenVoiceToneColorConverter")
class OpenVoiceToneColorConverter(BaseModel):
    def __init__(self,
                 ckpt_converter_path,
                 ckpt_whisper_path,
                 water_mark=None,
                 is_train=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.ckpt_converter_path = ckpt_converter_path
        self.ckpt_whisper_path = ckpt_whisper_path
        config_path = os.path.join(ckpt_converter_path, "converter", "config.json")
        hps = self.get_hparams_from_file(config_path)

        model = OpenVoiceSoVITS(
            len(getattr(hps, 'symbols', [])),
            hps.data.filter_length // 2 + 1,
            segment_size=None,
            n_speakers=hps.data.n_speakers,
            num_tones=None,
            num_languages=None,
            use_transformer_flow=False,
            **hps.model,
        ).to(self.device)


        self.model = model
        self.hps = hps
        self.watermark = water_mark
        self.load_ckpt(f'{ckpt_converter_path}/converter/checkpoint.pth')

        if water_mark is not None:
            import wavmark
            self.watermark_model = wavmark.load_model().to(self.device)
        else:
            self.watermark_model = None

        if is_train:
            self.model.train()
        else:
            self.model.eval()

    def load_ckpt(self, ckpt_path):
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        a, b = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        print("Loaded checkpoint '{}'".format(ckpt_path))
        print('missing/unexpected keys:', a, b)

    def extract_se(self, ref_wav_list, se_save_path=None):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]

        device = self.device
        hps = self.hps
        gs = []

        for fname in ref_wav_list:
            audio_ref, sr = librosa.load(fname, sr=hps.data.sampling_rate)
            y = torch.FloatTensor(audio_ref)
            y = y.to(device)
            y = y.unsqueeze(0)
            y = spectrogram_torch(y, hps.data.filter_length,
                                  hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                  center=False).to(device)
            with torch.no_grad():
                g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                gs.append(g.detach())
        gs = torch.stack(gs).mean(0)

        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(gs.cpu(), se_save_path)

        return gs

    def forward(self,
                audio,
                references_path,
                speaker,
                output_path=None,
                tau=0.3,
                source_sr=None,
                *args,
                **kwargs):
        hps = self.hps
        target_se, audio_name = speech_embedding.get_se(references_path, self, whisper_model=self.ckpt_whisper_path)
        # load audio
        if source_sr is not None:
            audio = librosa.resample(audio, source_sr, hps.data.sampling_rate, fix=True, scale=False)
        audio = torch.from_numpy(audio).float()

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            speaker_key = speaker.lower().replace('_', '-')
            src_se = torch.load(f"{self.ckpt_converter_path}/base_speakers/ses/{speaker_key}.pth", map_location=self.device)
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
