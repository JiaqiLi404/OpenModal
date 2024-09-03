from openmodal.engine import FlowBase


@FlowBase.register_module(name="BaseFlow")
class BaseFlow:
    def __init__(self, model: object):
        self._model = model

    def run(self):
        res=[]
        data=[5,10,15]
        for d in data:
            res.append(self.forward(d))
        return res

    def forward(self,input:any) -> any:
        ans = self._model.forward(input)
        print(ans)
        return ans
