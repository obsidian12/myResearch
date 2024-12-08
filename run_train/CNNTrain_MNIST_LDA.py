import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torchvision.transforms as transforms
import PIL
import module.pcaModule as wp
from glob import glob
import trainer

seedList = [6363, 8052, 5299, 2170, 7718, 6365, 4534, 1756, 542, 6032, 7163, 7182, 4204, 3638, 6946, 2892, 1542, 8704, 8465, 9348]
dimList = [30, 50, 100]
initFlagList = [True, False]

learning_rate = 0.01
training_epochs = 100
batch_size = 100
test_size = 100
itemNum = 10
sampleNum = 200

costMultiplier = 5
accMultiplier = 30

frozenFlag = False
# frozenStep = 0
# frozenDimList = [0, 1, 2, 3, 4]
# frozenBuffer = um.FrozenBuffer(d, 20, 0.15)
# frozenFlagContainer = []
# for i in range(d):
# 	frozenFlagContainer.append(False)

colorList = ["blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory",
             "blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory",
             "blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory"]

train_data = []
train_label = []
tf = transforms.ToTensor()

totalDataNum = 0

for i in range(10):
	print('C:/Users/dmtsa/OneDrive/문서/GitHub/weightPCA/DB/mnist_png/training/{}/*.png'.format(i))
	dataPathList = glob('C:/Users/dmtsa/OneDrive/문서/GitHub/weightPCA/DB/mnist_png/training/{}/*.png'.format(i))
	
	train_label_raw = []
	for j in range(10):
		train_label_raw.append([0])
	train_label_raw[i][0] = 1

	dataNum = len(dataPathList)
	totalDataNum = totalDataNum + dataNum
	for j in range(dataNum):
		train_data.append(tf(PIL.Image.open(dataPathList[j])))
		train_label.append(torch.FloatTensor(train_label_raw))

test_data = []
test_label = []

for i in range(10):
	dataPathList = glob('C:/Users/dmtsa/OneDrive/문서/GitHub/weightPCA/DB/mnist_png/testing/{}/*.png'.format(i))
	
	test_label_raw = []
	for j in range(10):
		test_label_raw.append([0])
	test_label_raw[i][0] = 1
	
	dataNum = len(dataPathList)
	for j in range(dataNum):
		test_data.append(tf(PIL.Image.open(dataPathList[j])))
		test_label.append(torch.FloatTensor(test_label_raw))

print("data importing process is complete!")

dayNum = 130
rsNum = 2
_, x = wp.makeDataArray(["./result_dldr/{:04d}{:02d}_otb_total_d{:02d}.csv".format(dayNum, rsNum, 400)])

print("basis importing process is complete!")

trainCounter = 2100
for initFlag in initFlagList:
	for dim in dimList:
		for seed in seedList:
			trainCounter = trainCounter + 1
			print("{}th training will be executed!".format(trainCounter))
			trainer.training_MNIST(trainCounter, seed, learning_rate, training_epochs, batch_size, test_size, itemNum, sampleNum, dim,
			 totalDataNum, train_data, test_data, train_label, test_label, x, initFlag, frozenFlag, colorList, accMultiplier, costMultiplier, 1, None)