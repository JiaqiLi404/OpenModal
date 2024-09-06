import os.path
from typing import Union, Optional
from tqdm import tqdm
import torch

from openmodal.engine import ModelBase
from openmodal.flow import BaseFlow
from openmodal.process.audio import speech_embedding


@ModelBase.register_module(name="TTSToneColorConverterFlow")
class TTSToneColorConverterFlow(BaseFlow):
    def __init__(self, tts_flow, converter_model, whisper_model_dir, output_dir: Union[str],
                 output_format: Optional[str] = 'wav', *args, **kwargs):
        """
        VoiceToneColorConverterFlow
        :param tts_flow: The TTS flow for generating general speech
        :param converter_model: The model for converting the tone color
        :param whisper_model_dir: The whisper model directory
        :param output_dir: The output directory
        :param output_format: The output format
        :param device:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.tts_flow = tts_flow
        self.converter_model = converter_model
        self.whisper_model_dir = whisper_model_dir
        self.output_dir = output_dir
        self.output_format = output_format

        self.converter_ckpt = self.converter_model.converter_ckpt

    def run(self):
        texts = [
            "使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开。开启后无视填写的参考文本。"
        ]
        references = [os.path.join('reference', "reference.m4a")]

        speaker_ids = self.tts_flow.tts_model.hps.data.spk2id
        if self.tts_flow.speaker is None or self.tts_flow.speaker not in speaker_ids:
            self.tts_flow.speaker = list(speaker_ids.keys())[0]
        self.tts_flow._speaker_id = speaker_ids[self.tts_flow.speaker]

        if len(references) != 1 and len(references) != len(texts):
            raise ValueError("The number of references should be 1 or equal to the number of tmp files.")
        if len(references) == 1:
            references = references * len(texts)

        os.makedirs(self.output_dir, exist_ok=True)

        self.converter_model.load_ckpt(f'{self.converter_ckpt}/converter/checkpoint.pth')
        for i, text in tqdm(enumerate(texts)):
            if self.tts_flow is not None:
                input = self.tts_flow.forward(text)
            else:
                input = text
            self.forward(input, references[i], str(i))

        return self.output_dir

    def forward(self, input, reference, output_filename):
        target_se, audio_name = speech_embedding.get_se(reference, self.converter_model,
                                                        whisper_model=self.whisper_model_dir)
        output_dir = f"{self.output_dir}/{output_filename}-output.{self.output_format}"
        if os.path.exists(output_dir):
            os.remove(output_dir)

        speaker_key = self.tts_flow.speaker.lower().replace('_', '-')
        self.converter_model.forward(
            input,
            src_se=torch.load(f"{self.converter_ckpt}/base_speakers/ses/{speaker_key}.pth", map_location=self.device),
            tgt_se=target_se,
            output_path=output_dir,
            source_sr=self.tts_flow.tts_model.hps.data.sampling_rate,
        )
