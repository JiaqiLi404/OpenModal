import os.path
from typing import Union, Optional
from tqdm import tqdm

from openmodal.engine import ModelBase
from openmodal.flow import BaseFlow
from openmodal.view_object.text.languages import LanguagesEnum


@ModelBase.register_module(name="TTSToneColorConverterFlow")
class TTSToneColorConverterFlow(BaseFlow):
    def __init__(self, tts_flow, converter_model, output_path: Union[str],
                 output_format: Optional[str] = 'wav', *args, **kwargs):
        """
        VoiceToneColorConverterFlow
        :param tts_flow: The TTS flow for generating general speech
        :param converter_model: The model for converting the tone color
        :param output_path: The output directory
        :param output_format: The output format
        :param device:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.tts_flow = tts_flow
        self.converter_model = converter_model
        self.output_path = output_path
        self.output_format = output_format

    def run(self):
        texts = [
            "使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开。开启后无视填写的参考文本。Hello every body!"
        ]
        references_path = [os.path.join('reference', "reference.m4a")]
        reference_text = [
            "一场台风过后，晴空万里。在离城市不远的近郊，有一个村庄遭到了台风的破坏。不过，损失还不太严重，仅仅是村外山脚下那座小小的庙被台风连根端跑了，并没有伤什么人。第二天早晨，村里人知道了这件事以后便纷纷议论起来。"]
        reference_language = [LanguagesEnum.ZH]

        if self.tts_flow is not None:
            speaker_ids = self.tts_flow.tts_model.hps.data.spk2id
            if self.tts_flow.speaker is None or self.tts_flow.speaker not in speaker_ids:
                self.tts_flow.speaker = list(speaker_ids.keys())[0]
            self.tts_flow._speaker_id = speaker_ids[self.tts_flow.speaker]

        if len(references_path) != 1 and len(references_path) != len(texts):
            raise ValueError("The number of references should be 1 or equal to the number of tmp files.")
        if len(references_path) == 1:
            references_path = references_path * len(texts)
        if len(reference_text) == 1:
            reference_text = reference_text * len(texts)
        if len(reference_language) == 1:
            reference_language = reference_language * len(texts)

        os.makedirs(self.output_path, exist_ok=True)

        for i, text in tqdm(enumerate(texts)):
            if self.tts_flow is not None:
                input = self.tts_flow.forward(text)
            else:
                input = text
            self.forward(input, references_path[i], reference_text[i], reference_language[i], str(i))

        return self.output_path

    def forward(self, input, references_path, reference_text, reference_language, output_filename):
        output_path = f"{self.output_path}/{output_filename}-output.{self.output_format}"
        if os.path.exists(output_path):
            os.remove(output_path)

        self.converter_model.forward(
            input,
            references_path=references_path,
            reference_text=reference_text,
            reference_language=reference_language,
            speaker=self.tts_flow.speaker if self.tts_flow is not None else None,
            output_path=output_path,
            source_sr=self.tts_flow.tts_model.hps.data.sampling_rate if self.tts_flow is not None else None,
        )
