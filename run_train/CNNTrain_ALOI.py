import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random as rd
import numpy as np
import PIL
from tqdm import tqdm
from module import utilModule as um
from module import netModuleCNN as nmc
from datetime import datetime
import matplotlib.pyplot as plt
import math
import module.pcaModule as wp

def training_aloi(rdSeed, showFlag):
	######################################################
	######################################################
	# 0. get random seed and save number from command line
	#rdSeed = int(sys.argv[1])

	torch.manual_seed(rdSeed)
	rd.seed(rdSeed)

	###################################################
	###################################################
	# 1-1. set learning parameter and prepare save folder
	learning_rate = 0.001
	training_epochs = 20
	batch_size = 10
	test_size = 100
	itemNum = 50
	sampleNum = 360

	costMultiplier = 1
	accMultiplier = 100

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

	###################################################
	###################################################
	# 1-2. prepare save folder

	#runDate = datetime.today().strftime("%Y_%m_%d")
	#prefix_tr = runDate + "_" +  str(saveNum).zfill(3) + "_tr"
	prefix = "3-4-8-16-fc__" + str(rdSeed).zfill(4)

	if not os.path.isdir("./" + prefix):
			os.mkdir("./" + prefix)

	###########################################################
	###########################################################
	# 2. import ALOI dataset as training data and testing data
	train_data = []
	tf = transforms.ToTensor()
	for objNum in range(1, itemNum + 1):
		for dirNum in range(36):
			fileName = "C:/Users/dmtsa/OneDrive/바탕 화면/연구/DB/aloi_view_png/{}/{}_r{}.png".format(objNum, objNum, dirNum * 10)
			train_data.append(tf(PIL.Image.open(fileName)))
		print("(training) No.{} object importing is complete.".format(objNum))

	test_data = []
	tf = transforms.ToTensor()
	for objNum in range(1, itemNum + 1):
		for dirNum in range(36):
			fileName = "C:/Users/dmtsa/OneDrive/바탕 화면/연구/DB/aloi_view_png/{}/{}_r{}.png".format(objNum, objNum, dirNum * 10 + 5)
			test_data.append(tf(PIL.Image.open(fileName)))
		print("(testing) No.{} object importing is complete.".format(objNum))

	label = []
	for objNum in range(itemNum):
		label_raw = []
		for i in range(itemNum):
			label_raw.append([0])
		
		label_raw[objNum][0] = 1

		for dirNum in range(36):
			label.append(torch.FloatTensor(label_raw))

	print("data importing process is complete!")


	###########################################################
	###########################################################
	# 3. select model, loss function and optimizer for training
	model = nmc.CNN_ALOI(itemNum)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	paramNum = 0
	for param in model.parameters():
		paramNum = paramNum + 1

	#########################################
	# 4. process raw data into pytorch tensor
	X_test_raw, Y_test_raw = um.sample(test_data, label, test_size, 36 * itemNum)
	X_test = torch.stack(X_test_raw, dim=0)
	Y_test = torch.stack(Y_test_raw, dim=0)
	Y_test = Y_test.squeeze()
	Y_test = torch.argmax(Y_test, 1)

	flat = nn.Flatten(start_dim=0)


	##########################
	# 5. execute training step
	costContainer = []
	accContainer = []
	trainAccContainer = []
	paramNormContainer = [] # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
	gradNormContainer = []

	sampler = um.sampler()
	totalDataNum = 36 * itemNum
	iterNum = int(totalDataNum / batch_size)
	period = int(iterNum * (training_epochs / sampleNum))

	for i in range(paramNum):
		paramNormContainer.append([])
		gradNormContainer.append([])

	for epoch in tqdm(range(training_epochs)):
		avg_cost = 0
		sampler.setBuffer(totalDataNum)

		for i in range(iterNum):

			x_raw, y_raw = sampler.sample(train_data, label, batch_size)

			X = torch.stack(x_raw, dim=0)
			Y = torch.stack(y_raw, dim=0)
			Y = Y.squeeze()

			optimizer.zero_grad()
			prediction = model(X)
			cost = criterion(prediction, Y)

			cost.backward()

			if i%(period) == 0:

				if i == 0:
					with torch.no_grad():
						Y_tmp = torch.argmax(Y, 1)
						result = torch.argmax(prediction, 1) == Y_tmp
						accurancy = result.float().mean()
						trainAccContainer.append(accurancy.detach().item())

				counter = 0
				for param in model.parameters():
					tmpGrad = flat(param.grad.detach()).tolist()
					
					gradNormContainer[counter].append(np.linalg.norm(np.array(tmpGrad))) # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
					counter = counter + 1

					fp = open("./" + prefix + "/layer{}_grad.csv".format(counter), 'a')
					for j in range(len(tmpGrad)):
						if j != 0: fp.write(",")
						fp.write(str(tmpGrad[j]))
					fp.write("\n")
					fp.close()

			optimizer.step()

			avg_cost = avg_cost + cost / iterNum

			if i%(period) == 0:

				counter = 0
				for param in model.parameters():
					tmp = flat(param)
					tmp = tmp.detach().tolist()
					paramNormContainer[counter].append(np.linalg.norm(np.array(tmp))) # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
					counter = counter + 1

					# save each parameter elements in divided csv file
					partitionNum = math.ceil(len(tmp) / 1000)
					for m in range(partitionNum):
						if not os.path.isdir("./" + prefix + "/layer{}".format(counter)):
							os.mkdir("./" + prefix + "/layer{}".format(counter))
						f = open("./" + prefix + "/layer{}/part{:04d}.csv".format(counter, m), 'a')
						if m < partitionNum - 1:
							for n in range(1000):
								if n != 0: f.write(",")
								f.write(str(tmp[m * 1000 + n]))
							f.write("\n")
							f.close()
						elif m == partitionNum - 1:
							for n in range(len(tmp) - 1000 * m):
								if n != 0: f.write(",")
								f.write(str(tmp[m * 1000 + n]))
							f.write("\n")
							f.close()

		costContainer.append(avg_cost.detach().item())
		print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

		with torch.no_grad():
			prediction = model(X_test)
			result = torch.argmax(prediction, 1) == Y_test
			accurancy = result.float().mean()
			accContainer.append(accurancy.detach().item())
			print('Accurancy : ', accurancy.item())


	###############################
	# 6. visualize and save results


	#paramSpace = np.linspace(1, training_epochs, sampleNum)
	testSpace = np.linspace(1, training_epochs, training_epochs)

	cost = np.array(costContainer) * costMultiplier
	acc = np.array(accContainer) * accMultiplier
	trainAcc = np.array(trainAccContainer) * accMultiplier

	plt.plot(testSpace, trainAcc, color=colorList[-4], label="Training accurancy")
	plt.plot(testSpace, acc, color=colorList[-2], label="Validation accurancy")
	plt.yticks(np.arange(0, 110, 10))
	plt.xticks(np.arange(1, training_epochs + 1))
	plt.ylabel("Accurancy")
	plt.xlabel("Epoch")
	plt.legend()
	plt.grid()
	plt.savefig(prefix + "/testResult.png")
	if showFlag: plt.show()

	# for i in range(paramNum):
	# 	plt.plot(paramSpace, paramNormContainer[i], color=colorList[i], label="norm of layer{}".format(i))
	# plt.plot(testSpace, cost, color=colorList[-3], label="cost[X{}]".format(costMultiplier))
	# plt.plot(testSpace, acc, color=colorList[-2], label="acc[X{}]".format(accMultiplier))
	# plt.legend()
	# plt.grid()
	# plt.title("cost, acc, and parameter norm")
	# plt.savefig(prefix_sd + "/testResult_paramNorm.png")
	# plt.show()
	# plt.clf()

	# for i in range(paramNum):
	# 	plt.plot(paramSpace, gradNormContainer[i], color=colorList[i], label="gradient norm of layer{}".format(i))
	# plt.plot(testSpace, cost, color=colorList[-3], label="cost[X{}]".format(costMultiplier))
	# plt.plot(testSpace, acc, color=colorList[-2], label="acc[X{}]".format(accMultiplier))
	# plt.legend()
	# plt.grid()
	# plt.title("cost, acc, and gradient norm")
	# plt.savefig(prefix_sd + "/testResult_gradNorm.png")
	# plt.show()
	# plt.clf()