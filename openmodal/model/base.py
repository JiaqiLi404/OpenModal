import torch
from openmodal.engine import ModelBase


@ModelBase.register_module(name="BaseModel")
class BaseModel(torch.nn.Module):
    def __init__(self, num=None, device='auto', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num = num

        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()
        self.device=device

    def forward(self, input_data: int) -> any:
        return self._num+input_data
