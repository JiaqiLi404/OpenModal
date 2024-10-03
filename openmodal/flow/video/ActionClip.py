from typing import Union, List, Optional
import os

from openmodal.engine import FlowBase
from openmodal.flow import BaseFlow
from openmodal.model import BaseModel

@FlowBase.register_module(name="ActionRecognitionFlow")
class ActionRecognitionFlow(BaseFlow):
    def __init__(self,  *args,**kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        res = []
        data = [5, 10, 15]
        for d in data:
            res.append(self.forward(d))
        return res

    def forward(self, input: any) -> any:
        ans = self.model.forward(input)
        print(ans)
        return ans
