import os.path
from typing import Union, Optional
from tqdm import tqdm
import torch

from openmodal.engine import ModelBase
from openmodal.process.audio import speech_embedding


@ModelBase.register_module(name="TTSToneColorConverterFlow")
class TTSToneColorConverterFlow:
    def __init__(self, tts_flow, converter_model, whisper_model_dir, output_dir: Union[str],
                 output_format: Optional[str] = 'wav', clean_cache=True, device='cuda',
                 *args, **kwargs):
        """
        VoiceToneColorConverterFlow
        :param tts_flow: The TTS flow for generating general speech
        :param converter_model: The model for converting the tone color
        :param whisper_model_dir: The whisper model directory
        :param output_dir: The output directory
        :param output_format: The output format
        :param clean_cache: Whether to clean the cache of the TTS flow
        :param device:
        :param args:
        :param kwargs:
        """
        self.tts_flow = tts_flow
        self.converter_model = converter_model
        self.whisper_model_dir = whisper_model_dir
        self.output_dir = output_dir
        self.output_format = output_format
        self.clean_cache = clean_cache
        self.device = device

        self.tmp_dir = self.tts_flow.output_dir
        self.tmp_format = self.tts_flow.output_format
        self.converter_ckpt = self.converter_model.converter_ckpt

    def run(self):
        self.tts_flow.run()
        os.makedirs(self.output_dir, exist_ok=True)

        tmp_files = [f"{self.tmp_dir}/{x}" for x in os.listdir(self.tmp_dir) if x.endswith(self.tmp_format)]
        references = [os.path.join('reference', "reference.m4a")]
        if len(references) != 1 and len(references) != len(tmp_files):
            raise ValueError("The number of references should be 1 or equal to the number of tmp files.")
        if len(references) == 1:
            references = references * len(tmp_files)

        self.converter_model.load_ckpt(f'{self.converter_ckpt}/converter/checkpoint.pth')
        for i, file in tqdm(enumerate(tmp_files)):
            self.forward(file, references[i])

        if self.clean_cache:
            for file in tmp_files:
                os.remove(file)

        return self.output_dir

    def forward(self, input_dir, reference):
        target_se, audio_name = speech_embedding.get_se(reference, self.converter_model,
                                                        whisper_model=self.whisper_model_dir)
        speaker_key = self.tts_flow.speaker.lower().replace('_', '-')
        filename=os.path.basename(input_dir)
        filename =os.path.splitext(filename)[0]
        self.converter_model.convert(
            audio_src_path=input_dir,
            src_se=torch.load(f"{self.converter_ckpt}/base_speakers/ses/{speaker_key}.pth", map_location=self.device),
            tgt_se=target_se,
            output_path=f"{self.output_dir}/{filename}-output.{self.output_format}",
        )
