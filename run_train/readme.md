# run_train

## about trainer.py file
There are modelLoader classes(UnetLoader, Resnet18Loader, Resnet18SmallLoader, VGG16Loader, ...)

and trainer classes(CIFAR10Trainer, ...) in trainer.py

### when using modelLoader classes

Since these classes are mainly used by trainer classes, so you don't need to how to use these classes

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

### when implementing new trainer classes
