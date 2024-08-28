from typing import Dict, Any, Callable
import copy
import re
from tqdm import tqdm

from openmodal.engine import Config
from openmodal.engine import Registry, MainBase, FlowBase, ModelBase, BlockBase, MetricBase, DatasetBase, ProcessBase, \
    PreProcessBase, ViewBase, VisualizationBase

config = Config.fromfile('config/example.yaml')

# sort the modules if possible
pre_modules = []
runtime_modules = []
post_modules = []
for key, value in config.items():
    if isinstance(value, dict) and "order" in value:
        if value["order"] < 0:
            pre_modules.append(key)
        else:
            post_modules.append(key)
    else:
        runtime_modules.append(key)
pre_modules.sort(key=lambda x: config[x]["order"])
post_modules.sort(key=lambda x: config[x]["order"])

# print the modules in order
print("Your Configurations:")
for module in pre_modules:
    print(f"{module}: {config[module]}")
for module in runtime_modules:
    print(f"{module}: {config[module]}")
for module in post_modules:
    print(f"{module}: {config[module]}")
print("\n\n")

# build the modules
global_built_modules = {}


def build_module(name: str, cfg: Any, outer_built_modules: Dict[str, Any]) -> Any:
    """
    Build the module according to the configuration.
    Also, we support recognizing the variables in the configuration and replace the {{XXX}} in configurations to the built modules.
    :param name: item name
    :param cfg: item configuration
    :param outer_built_modules: outer variable scope
    :return: built module
    """
    private_built_modules = copy.deepcopy(outer_built_modules)
    rule = re.compile(r"\{\{[a-zA-Z0-9]+}}")

    if isinstance(cfg, dict) and 'type' in cfg:
        for key, value in cfg.items():
            if isinstance(value, dict) and 'type' in value:
                cfg[key] = build_module(key, value, private_built_modules)
            if isinstance(value, str) and rule.match(value):
                inner_name = value[2:-2].strip()
                if inner_name in private_built_modules:
                    cfg[key] = private_built_modules[inner_name]
                else:
                    raise KeyError(f"Variable {inner_name} not found in the built modules, "
                                   f"please check whether the variable is defined before it is used.")
        module = MainBase.build(cfg)
    elif isinstance(cfg, str) and rule.match(cfg):
        inner_name = cfg[2:-2].strip()
        if inner_name in private_built_modules:
            module = private_built_modules[inner_name]
        else:
            raise KeyError(f"Variable {inner_name} not found in the built modules, "
                           f"please check whether the variable is defined before it is used.")
    else:
        module = cfg

    outer_built_modules[name] = module
    return cfg


print("Your Final Configurations:")
for name in tqdm(pre_modules, desc="Building pre-modules"):
    cfg = config[name]
    cfg = build_module(name, cfg, global_built_modules)
    print(f"{name}: {cfg}")
for name in tqdm(runtime_modules, desc="Building runtime-modules"):
    cfg = config[name]
    cfg = build_module(name, cfg, global_built_modules)
    print(f"{name}: {cfg}")
for name in tqdm(post_modules, desc="Building post_modules"):
    cfg = config[name]
    cfg = build_module(name, cfg, global_built_modules)
    print(f"{name}: {cfg}")
print("\n\n")

flow = global_built_modules["flow"]
flow.forward()
