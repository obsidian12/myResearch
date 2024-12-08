import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torchvision.transforms as transforms
import PIL
import module.pcaModule as wp
import trainer

frozenFlag = False
# frozenStep = 0
# frozenDimList = [0, 1, 2, 3, 4]
# frozenBuffer = cm.FrozenBuffer(d, 20, 0.15)
# frozenFlagContainer = []
# for i in range(d):
# 	frozenFlagContainer.append(False)

seedList = [6363]
			#, 8052, 5299, 2170, 7718, 6365, 4534, 1756, 542, 6032, 7163, 7182, 4204, 3638, 6946, 2892, 1542, 8704, 8465, 9348]
#dimList = [20, 30, 50]
dimList = [30]
initFlagList = [True, False]

learning_rate = 0.005
training_epochs = 100
batch_size = 50
test_size = 100
item_num = 50
sampling_num = 200

costMultiplier = 0.00001
accMultiplier = 30
gradNormMultiplier = 1

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
tf = transforms.ToTensor()
for objNum in range(1, item_num + 1):
	for dirNum in range(36):
		fileName = "C:/Users/dmtsa/OneDrive/문서/GitHub/weightPCA/DB/aloi_view_png/{}/{}_r{}.png".format(objNum, objNum, dirNum * 10)
		train_data.append(tf(PIL.Image.open(fileName)))
	print("(training) No.{} object importing is complete.".format(objNum))

test_data = []
tf = transforms.ToTensor()
for objNum in range(1, item_num + 1):
	for dirNum in range(36):
		fileName = "C:/Users/dmtsa/OneDrive/문서/GitHub/weightPCA/DB/aloi_view_png/{}/{}_r{}.png".format(objNum, objNum, dirNum * 10 + 5)
		test_data.append(tf(PIL.Image.open(fileName)))
	print("(testing) No.{} object importing is complete.".format(objNum))

label = []
for objNum in range(item_num):
	label_raw = []
	for i in range(item_num):
		label_raw.append([0])
	
	label_raw[objNum][0] = 1

	for dirNum in range(36):
		label.append(torch.FloatTensor(label_raw))

print("data importing process is complete!")


dayNum = 119
rsNum = 1
_, x = wp.makeDataArray(["./result_dldr/{:04d}{:02d}_otb_total_d{:02d}.csv".format(dayNum, rsNum, 50)])

print("basis importing process is complete!")

trainCounter = 0
for initFlag in initFlagList:
	for dim in dimList:
		for seed in seedList:
			trainCounter = trainCounter + 1
			print("{}th training will be executed!".format(trainCounter))
			trainer.training_ALOI(trainCounter, seed, learning_rate, training_epochs, batch_size, test_size, item_num, sampling_num, dim,
			 train_data, test_data, label, x, initFlag, frozenFlag, colorList, accMultiplier, costMultiplier, gradNormMultiplier)