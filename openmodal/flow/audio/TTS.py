from typing import Union, List, Optional
import os

from tqdm import tqdm

from openmodal.engine import FlowBase


@FlowBase.register_module(name="TTSFlow")
class TTSFlow:
    def __init__(self, tts_model: object, speaker: any, output_dir: Union[str], output_format='wav', device='cuda:0',
                 *args, **kwargs):
        """
        TTS Flow: Text to Speech audio flow, which is used to convert text to speech audio.
        :param tts_model: The TTS model object.
        :param speaker: The speaker id contained in the TTS model config file.
        :param output_dir: The output directory to save the generated audio.
        :param output_format: The output audio format, default is 'wav'.
        :param device: The device to run the TTS model, default is 'cuda:0'.
        """
        self.tts_model = tts_model
        self.speaker = speaker
        self.output_dir = output_dir
        self.output_format = output_format
        self.device = device

        self._speaker_id = None

    def run(self):
        texts = [
            "通过拖拽多个文件上传多个参考音频（建议同性），平均融合他们的音色。如不填写此项，音色由左侧单个参考音频控制。如是微调模型，建议参考音频全部在微调训练集音色内，底模不用管。"
        ]

        # create the output directory, clean it if it exists
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
        os.makedirs(self.output_dir, exist_ok=True)

        speaker_ids = self.tts_model.hps.data.spk2id
        if self.speaker is None or self.speaker not in speaker_ids:
            self.speaker = list(speaker_ids.keys())[0]
        self._speaker_id = speaker_ids[self.speaker]
        for text in texts:
            self.forward(text, "tmp")

    def generate_tmp_output_dir(self, prefix, nums) -> List[str]:
        # count the files in the output_dir
        start_index = len(os.listdir(self.output_dir))
        output_dir = [os.path.join(self.output_dir, f"{prefix}-{start_index + i}.{self.output_format}") for i in
                      range(nums)]
        return output_dir

    def forward(self, input: Union[str, List], output_filename: Optional[Union[str, List]], speed=1.0) -> any:
        # judging if input is List
        if isinstance(input, str):
            input = [input]
        if output_filename is None:
            output_dir = self.generate_tmp_output_dir("", len(input))
        elif isinstance(output_filename, str):
            output_dir = self.generate_tmp_output_dir(output_filename, len(input))
        elif isinstance(output_filename, list) and len(output_filename) != len(input):
            raise ValueError("The length of output_filename should be equal to the length of input.")
        for (inp, out) in tqdm(zip(input, output_dir), desc="TTS Flow"):
            self.tts_model.tts_to_file(inp, self._speaker_id, out, speed=speed)
