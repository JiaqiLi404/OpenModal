# OpenModal
## Introduction

OpenModal is a foundational library for training multi-modal models based on PyTorch. The highlights are:
- **Modular Design**: The library is designed in a modular way, which allows you to easily add and customize the components, just by registering them in the module registry.
- **Configuration**: The library supports configuration files in yaml and json formats, which allows you to easily assemble different models and manage the hyperparameters during your experiments.
- **Flexibility**: Due to the variety of multi-modal tasks, the library provides a flexible way to define the components and the flow of the project.
- **Standardized**: We recommend a standardized way to define the formal object-oriented model inputs and outputs, which could help others to easily understand and utilize the model.

## Supported Models:
- [ ] [MeloTTS](https://github.com/myshell-ai/MeloTTS) TTS Model

## Supporting Customizing Modules:
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
  model: "{{model}}"
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
A flow is necessary for the project, which defines the flow of the project, including data loading, preprocessing, augmentation, inferencing, etc.
A correct configuration should define the necessary modules first, and include the `flow` item to control the data flow between the models.
e.g.:
```yaml
depth: 18
num_classes: 10
model:
  type: "ResNet"
  num_classes: "{{num_classes}}"
  depth: "{{depth}}"
image_loader:
  type: "ImageLoader"
  root: "data\image"
flow:
  type: "BaseFlow"
  model: "{{model}}"
  audio_loader:
    type: "AudioLoader"
    root: "data\audio"
  image_loader: "{{image_loader}}"
```