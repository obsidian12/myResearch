from trainer import *

tr = CIFAR10Trainer()
tr.training("vgg16", 1827, "./config_vgg16.json")