import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dict = unpickle('C:/Users/dmtsa/RESEARCH/DB/cifar-10-batches-py/data_batch_1')
print(dict[b'data'].shape)
print(len(dict[b'labels']))

# a = dict[b'data'][1]
# image = np.reshape(a, (3, 32, 32))
# image = np.swapaxes(image, 0, 2)
# image = np.swapaxes(image, 0, 1)
# plt.imshow(image)
# plt.show()
yeah = torch.tensor(dict[b'labels'], dtype=torch.int64)
print(yeah)
train_label = F.one_hot(yeah, num_classes=10)
print(train_label.shape)