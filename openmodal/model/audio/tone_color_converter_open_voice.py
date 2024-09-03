import torch
import torch.nn as nn
import numpy as np
import soundfile
import os
import librosa

from openmodal.engine import ModelBase
from openmodal.model.audio.TTS import SynthesizerTrn, get_hparams_from_file


@ModelBase.register_module(name="OpenVoiceToneColorConverter")
class OpenVoiceToneColorConverter(nn.Module):
    def __init__(self, converter_ckpt, water_mark=None, device='cuda:0', *args, **kwargs):
        super().__init__()
        if 'cuda' in device:
            assert torch.cuda.is_available()

        self.converter_ckpt = converter_ckpt
        config_path = os.path.join(converter_ckpt, "converter", "config.json")
        hps = get_hparams_from_file(config_path)

        model = SynthesizerTrn(
            len(getattr(hps, 'symbols', [])),
            hps.data.filter_length // 2 + 1,
            segment_size=None,
            n_speakers=hps.data.n_speakers,
            num_tones=None,
            num_languages=None,
            use_transformer_flow=False,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.hps = hps
        self.device = device
        self.watermark = water_mark

        if water_mark is not None:
            import wavmark
            self.watermark_model = wavmark.load_model().to(self.device)
        else:
            self.watermark_model = None

    def load_ckpt(self, ckpt_path):
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        a, b = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        print("Loaded checkpoint '{}'".format(ckpt_path))
        print('missing/unexpected keys:', a, b)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        return input

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

    def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3):
        hps = self.hps
        # load audio
        audio, sample_rate = librosa.load(audio_src_path, sr=hps.data.sampling_rate)
        audio = torch.tensor(audio).float()

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            audio = self.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][
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
