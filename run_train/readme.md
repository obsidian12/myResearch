# run_train

this folder is for training model, and saving weights of model in a given format.

you can get your train result and weight trajectories as a directory(`{model}__{seeds}`).

## about trainer.py file
There are modelLoader classes(UnetLoader, Resnet18Loader, Resnet18SmallLoader, VGG16Loader, ...)

and trainer classes(CIFAR10Trainer, ...) in trainer.py

### when using modelLoader classes

Since these classes are mainly used by trainer classes, so you don't need to know how to use these classes

* `.get_model()` : you can get model which modelLoader creates.
* `.save_weights()` : you can save model weights which modelLoader has.

### when implementing new modelLoader classes

* `_build_new_model()`
    - input : None(only self)
    - output : None
    - function : need to create new model instance, and save it to self.model, you can use self.num_classes information
* `_build_non_save_layers()`
    - input : None(only self)
    - output : None
    - function : need to definite self.non_save_layers
    - `self.non_save_layers` : list of strings. when save weights, layers having name which includes elements of `self.non_save_layers` are not saved  
* additionally, you need to register your new modelLoader class to `trainer.load_model()` methods

### when using trainer classes

* `training(model_name, rdSeed, config_file_path)` : you can create model, train created model, and save results
    - `model_name` : string, model name to use, Regardless of case
    - `rdSeed` : int, random seed for random, pytorch modules
    - `config_file_path` : string, youre config file's path, see examples of config files in this directory

### when implementing new trainer classes

* `load_DB()`
    - input : None(only self)
    - output : tuple of train DataLoader, test DataLoader
    - function : need to load databases, transform databases to pytorch Tensor, and build and return Dataloaders for training, testing. you can use self.batch_size information
* `build_num_classes()`
    - input : None(only self)
    - output : None
    - function : save number of classes of your databases to self.num_classes
