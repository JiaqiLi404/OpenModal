from openmodal.engine import ModelBase


@ModelBase.register_module(name="BaseModel")
class BaseModel:
    def __init__(self, num:int):
        self._num = num

    def forward(self, input_data: int) -> any:
        return self._num+input_data
