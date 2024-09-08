from typing import Union, List, Optional
import os

from openmodal.engine import FlowBase
from openmodal.flow import BaseFlow
from openmodal.model import BaseModel


@FlowBase.register_module(name="TTSFlow")
class TTSFlow(BaseFlow):
    def __init__(self, tts_model: BaseModel, speaker: any, output_path: Union[str], output_format='wav', *args,
                 **kwargs):
        """
        TTS Flow: Text to Speech audio flow, which is used to convert text to speech audio.
        :param tts_model: The TTS model object.
        :param speaker: The speaker id contained in the TTS model config file.
        :param output_path: The output directory to save the generated audio.
        :param output_format: The output audio format, default is 'wav'.
        :param device: The device to run the TTS model, default is 'cuda:0'.
        """
        super().__init__(*args, **kwargs)
        self.tts_model = tts_model
        self.speaker = speaker
        self.output_path = output_path
        self.output_format = output_format

        self._speaker_id = None

    def run(self):
        texts = [
            "使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开。开启后无视填写的参考文本。"
        ]

        outputs = []

        # create the output directory, clean it if it exists
        if os.path.exists(self.output_path):
            for file in os.listdir(self.output_path):
                os.remove(os.path.join(self.output_path, file))
        os.makedirs(self.output_path, exist_ok=True)

        speaker_ids = self.tts_model.hps.data.spk2id
        if self.speaker is None or self.speaker not in speaker_ids:
            self.speaker = list(speaker_ids.keys())[0]
        self._speaker_id = speaker_ids[self.speaker]
        for text in texts:
            outputs.append(self.forward(text, "tmp"))

        return outputs

    def forward(self, input: str, output_filename: Optional[str] = None, speed=1.0) -> any:
        # judging if input is List
        output_path = None
        if output_filename is not None:
            output_path = os.path.join(self.output_path, f"{output_filename}.{self.output_format}")
            if os.path.exists(output_path):
                os.remove(output_path)
        output = self.tts_model.forward(input, self._speaker_id, output_path, speed=speed)
        return output
