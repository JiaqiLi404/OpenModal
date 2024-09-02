# @Time : 2023/9/23 23:26
# @Author : Li Jiaqi
# @Description : The registry for the modules
from typing import Dict, Type, Optional, Union, List, Callable, Any


class Registry:
    def __init__(self, name: str = "register", parent: Optional = None):
        self._modules_: Dict[str, Callable] = dict()
        self._name = name
        self._parent = parent
        self._child = []
        parent.add_child(self) if parent is not None else None

    def add_child(self, child):
        self._child.append(child)

    def _register_module(self, module: Type = None, module_name: Optional[Union[str, List[str]]] = None):
        if not callable(module):
            raise TypeError(
                f'module must be Callable, but got {type(module)}')
        if isinstance(module_name, str):
            module_name = [module_name]

        for name in module_name:
            if name is None:
                name = module.__name__
            if name in self._modules_:
                print("module is already registered in " + name)
            self._modules_[name] = module
        # print(f"Register {module_name} in {self._name}")

    def register_module(self,
                        name: Optional[Union[str, List[str]]] = None,
                        module: Optional[Type] = None
                        ) -> Union[Type, Callable]:
        if not (name is None or isinstance(name, str) or isinstance(name, list) and all(
                isinstance(n, str) for n in name)):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name)
            return module

        def _register(module: Type) -> Type:
            self._register_module(module=module, module_name=name)
            return module

        return _register

    def get(self, name: str) -> Optional[Type]:
        if name in self._modules_:
            return self._modules_.get(name)
        for child in self._child:
            obj = child.get(name)
            if obj is not None:
                return obj
        return None

    def build(self, cfg: Dict[str, Any]) -> Any:
        if not isinstance(cfg, dict) or 'type' not in cfg:
            raise TypeError('cfg must be a dict and contain the key "type"')
        module_type = cfg.pop('type').strip()
        module_cls = self.get(module_type)
        if module_cls is None:
            raise KeyError(f'{module_type} is not registered in {self._name}')
        return module_cls(**cfg)

    def __repr__(self):
        num = len(self._modules_)
        for child in self._child:
            num += child.__repr__()[1]
        return f"{self._name} Registry has {len(self._modules_)} modules.", num


MainBase = Registry("main")

FlowBase = Registry("flow", parent=MainBase)

ModelBase = Registry("model", parent=MainBase)

ModuleBase=Registry("module",parent=MainBase)

BlockBase = Registry("block", parent=MainBase)

MetricBase = Registry("metric", parent=MainBase)

DatasetBase = Registry("dataset", parent=MainBase)

ProcessBase = Registry("process", parent=MainBase)
PreProcessBase = Registry("preprocess", parent=ProcessBase)
PostProcessBase = Registry("postprocess", parent=ProcessBase)

ViewBase = Registry("view", parent=MainBase)

VisualizationBase = Registry("visualization", parent=MainBase)
