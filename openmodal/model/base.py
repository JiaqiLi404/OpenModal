import torch
from huggingface_hub import hf_hub_download
import os
import json
from openmodal.engine import ModelBase


@ModelBase.register_module(name="BaseModel")
class BaseModel(torch.nn.Module):
    def __init__(self, num=None, device='auto',is_half=None, *args, **kwargs):
        super().__init__()
        self._num = num

        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if device is not None and 'cuda' in device:
            assert torch.cuda.is_available()
        # AttributeError: property 'device' of object has no setter
        if device is not None:
            self.device = device

        if is_half:
            self.half()
        self.is_half=is_half


    def forward(self, input_data: int) -> any:
        return self._num + input_data

    def get_hparams_from_file(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = f.read()
        config = json.loads(data)

        hparams = HParams(**config)
        return hparams

    def load_or_download_model(self, ckpt_path, device, load_model=True):
        download_model, download_config = False, False
        model_path, config_path = None, None
        if not os.path.exists(ckpt_path):
            download_model, download_config = True, True
        else:
            if os.path.isfile(ckpt_path):
                files=[ckpt_path]
            else:
                files = os.listdir(ckpt_path)
            model_files = [f for f in files if f.endswith('.pth') or f.endswith('.ckpt')]
            if len(model_files) == 0:
                download_model = True
            else:
                model_path = os.path.join(ckpt_path, model_files[0])
            config_files = [f for f in files if f.endswith('.json')]
            if len(config_files) == 0 and (len(model_files) != 0 and not model_files[0].endswith('.ckpt')):
                download_config = True
            elif len(config_files) != 0:
                config_path = os.path.join(ckpt_path, config_files[0])
        if download_model:
            model_path = hf_hub_download(repo_id=ckpt_path, filename="checkpoint.pth") if load_model else None
        if download_config:
            config_path = hf_hub_download(repo_id=ckpt_path, filename="config.json")
        # loading both config and weight from the ckpt file
        if config_path is None:
            model = torch.load(model_path, map_location="cpu")
            config = model['config']
            config = HParams(**config)
            model = model['weight']
        else:
            model = torch.load(model_path, map_location=device) if load_model else None
            model = model['model'] if load_model and 'model' in model.keys() else None
            config = self.get_hparams_from_file(config_path)
        return model, config


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

    def get(self, key,default=None):
        """
        hps.get("n_layers")
        hps.get("n_layers", 3)
        hps.get(["n_layers","layers_num"], 3)

        :param key:
        :param default:
        :return:
        """
        if isinstance(key, list):
            for k in key:
                if k in self.__dict__:
                    return getattr(self, k)
            return default
        if key in self.__dict__:
            return getattr(self, key)
        return default

    def set(self, key, value):
        setattr(self, key, value)

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
