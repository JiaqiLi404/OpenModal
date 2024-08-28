from openmodal.engine import FlowBase


@FlowBase.register_module(name="BaseFlow")
class BaseFlow:
    def __init__(self, model: object):
        self._model = model

    def forward(self) -> any:
        ans = self._model.forward(5)
        print(ans)
        return ans
