from trainer import *

tr = CIFAR10Trainer()
tr.training("vgg16", 239, "./config.json")