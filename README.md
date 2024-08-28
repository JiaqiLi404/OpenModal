## Introduction

OpenModal is a foundational library for training multi-modal models based on PyTorch. The highlights are:
- **Registry**: The library provides a registry mechanism to manage the components, such as models, datasets, metrics, etc.
- **Modular Design**: The library is designed in a modular way, which allows you to easily customize the components.
- **Configuration**: The library supports configuration files in yaml and json formats, which allows you to easily manage the hyperparameters during your experiments.


## Supporting modules:
- **flow**: Here you could define the flow of the project, i.e. including data loading, preprocessing, augmentation, inferencing, etc.
- **model**: Here you could define the model architecture.
- **block**: Here you could define the building blocks of the model.
- **metric**: Here you could define the evaluation metrics.
- **dataset**: Here you could define the dataset loaders.
- **process**: Here you could define the additional processes, e.g. screening, augmenting, etc.
- **view_object**: Here you could define the formal object-oriented model inputs and outputs.
- **visualization**: Here you could define the visualization functions.

## Configs
The configuration files are stored in the `configs` directory, which support yaml and json formats.

In a config, we support you to define the variables and the modules. 
When defining the models, you could use the `type` key to specify the model type, which should be registered in the module registry.
For example, you could define a ResNet model by the following code:
```yaml
num_classes: 10
model:
  type: "ResNet"
  num_classes: "{{num_classes}}"
  depth: 18
```

When you are using the variables, you must use the `{{variable_name}}` format, 
and confirm that the variable is defined in the config before using it.
By defining the loading order of the config items, you could use the `order` key.
Every item has a default order of 0. 
e.g.:
```yaml
flow:
  type: "BaseFlow"
  input: "{{model}}"
model:
  order: -1
  type: "ResNet"
  num_classes: "{{num_classes}}"
  depth: 18
```
The usage of configuration is similar to and partially copied from the [mmengine](https://github.com/open-mmlab/mmengine/tree/main).
Thus, you could structure your configurations by providing `_base_`, to inherit the base configuration; and providing `_delete_: True"` to overwrite the base configuration.
e.g.:
```yaml
_base_: "base.yaml"
model:
  _delete_: True
  order: -1
  type: "ResNet"
  num_classes: "{{num_classes}}"
  depth: 18
```
