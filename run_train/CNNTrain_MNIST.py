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
from glob import glob

def training_mnist(rdSeed):

	######################################################
	######################################################
	# 0. get random seed and save number from command line

	torch.manual_seed(rdSeed)
	rd.seed(rdSeed)

	###################################################
	###################################################
	# 1-1. set learning parameter and prepare save folder
	learning_rate = 0.05
	training_epochs = 20
	batch_size = 100
	test_size = 100
	itemNum = 10
	sampleNum = 2000

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

	prefix_w = "1-2-2-4-fc__" + str(rdSeed).zfill(4)
	# prefix_grad = "1-2-2-4-fc__" + str(rdSeed).zfill(4) + "__grad"

	if not os.path.isdir("./" + prefix_w):
		os.mkdir("./" + prefix_w)
	# if not os.path.isdir("./" + prefix_grad):
	# 	os.mkdir("./" + prefix_grad)

	###########################################################
	###########################################################
	# 2. import MNIST dataset as training data and testing data
	train_data = []
	train_label = []
	tf = transforms.ToTensor()

	totalDataNum = 0

	for i in range(10):
		print('C:/Users/dmtsa/RESEARCH/DB/mnist_png/training/{}/*.png'.format(i))
		dataPathList = glob('C:/Users/dmtsa/RESEARCH/DB/mnist_png/training/{}/*.png'.format(i))
		
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
		dataPathList = glob('C:/Users/dmtsa/RESEARCH/DB/mnist_png/testing/{}/*.png'.format(i))
		
		test_label_raw = []
		for j in range(10):
			test_label_raw.append([0])
		test_label_raw[i][0] = 1
		
		dataNum = len(dataPathList)
		for j in range(dataNum):
			test_data.append(tf(PIL.Image.open(dataPathList[j])))
			test_label.append(torch.FloatTensor(test_label_raw))

	print("data importing process is complete!")
	print("totalDataNum : {}".format(totalDataNum))


	###########################################################
	###########################################################
	# 3. select model, loss function and optimizer for training
	model = nmc.CNN_MNIST(itemNum)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	a = 1
	for name, param in model.named_parameters():
		print("{:02d} : ".format(a))
		print(name,param.shape)
		a += 1

	paramNum = 0
	for param in model.parameters():
		paramNum = paramNum + 1

	#########################################
	# 4. process raw data into pytorch tensor
	X_test = torch.stack(test_data, dim=0)
	Y_test = torch.stack(test_label, dim=0)
	Y_test = Y_test.squeeze()
	Y_test = torch.argmax(Y_test, 1)

	flat = nn.Flatten(start_dim=0)


	##########################
	# 5. execute training step
	costContainer = []
	accContainer = []
	paramNormContainer = [] # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
	gradNormContainer = []

	sampler = um.sampler()
	iterNum = int(totalDataNum / batch_size)
	period = int(iterNum * (training_epochs / sampleNum))

	for i in range(paramNum):
		paramNormContainer.append([])
		gradNormContainer.append([])

	for epoch in tqdm(range(training_epochs)):
		avg_cost = 0
		sampler.setBuffer(totalDataNum)

		for i in range(iterNum):

			x_raw, y_raw = sampler.sample(train_data, train_label, batch_size)

			X = torch.stack(x_raw, dim=0)
			Y = torch.stack(y_raw, dim=0)
			Y = Y.squeeze()

			optimizer.zero_grad()
			prediction = model(X)
			cost = criterion(prediction, Y)

			cost.backward()

			# if i%(period) == 0:
			# 	counter = 0
			# 	for param in model.parameters():
			# 		tmpGrad = flat(param.grad.detach()).tolist()
					
			# 		gradNormContainer[counter].append(np.linalg.norm(np.array(tmpGrad))) # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
			# 		counter = counter + 1

			# 		fp = open("./" + prefix_w_sd + "/layer{}_grad.csv".format(counter), 'a')
			# 		for j in range(len(tmpGrad)):
			# 			if j != 0: fp.write(",")
			# 			fp.write(str(tmpGrad[j]))
			# 		fp.write("\n")
			# 		fp.close()

			optimizer.step()

			# avg_cost = avg_cost + cost / iterNum

			if i%(period) == 0:

				counter = 0
				for param in model.parameters(): # iteration : each parameter
					param_flatten = flat(param)
					param_flatten = param_flatten.detach().tolist()
					# paramNormContainer[counter].append(np.linalg.norm(np.array(param_flatten))) # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
					# grad_flatten = flat(param.grad)
					# grad_flatten = grad_flatten.detach().tolist()
					counter = counter + 1

					# save each parameter elements in divided csv file
					partitionNum = math.ceil(len(param_flatten) / 1000) 
					for m in range(partitionNum): # iteration : each divied partition in one parameter
						if not os.path.isdir("./" + prefix_w + "/layer{}".format(counter)):
							os.mkdir("./" + prefix_w + "/layer{}".format(counter))
						# if not os.path.isdir("./" + prefix_grad + "/layer{}".format(counter)):
						# 	os.mkdir("./" + prefix_grad + "/layer{}".format(counter))
						f = open("./" + prefix_w + "/layer{}/part{:04d}.csv".format(counter, m), 'a')
						if m < partitionNum - 1:
							for n in range(1000):
								if n != 0: f.write(",")
								f.write(str(param_flatten[m * 1000 + n]))
							f.write("\n")
							f.close()
						elif m == partitionNum - 1:
							for n in range(len(param_flatten) - 1000 * m):
								if n != 0: f.write(",")
								f.write(str(param_flatten[m * 1000 + n]))
							f.write("\n")
							f.close()

						# f = open("./" + prefix_grad + "/layer{}/part{:04d}.csv".format(counter, m), 'a')
						# if m < partitionNum - 1:
						# 	for n in range(1000):
						# 		if n != 0: f.write(",")
						# 		f.write(str(grad_flatten[m * 1000 + n]))
						# 	f.write("\n")
						# 	f.close()
						# elif m == partitionNum - 1:
						# 	for n in range(len(grad_flatten) - 1000 * m):
						# 		if n != 0: f.write(",")
						# 		f.write(str(grad_flatten[m * 1000 + n]))
						# 	f.write("\n")
						# 	f.close()
				costContainer.append(cost.detach().item())
				if i == (iterNum - period):
					print()
					print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, cost.detach().item()), end=" || ")
				
				with torch.no_grad():
					prediction = model(X_test)
					result = torch.argmax(prediction, 1) == Y_test
					accurancy = result.float().mean()
					accContainer.append(accurancy.detach().item())
					if i == (iterNum - period): print('Accurancy : ', accurancy.item())

		# costContainer.append(avg_cost.detach().item())
		# print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

		# with torch.no_grad():
		# 	prediction = model(X_test)
		# 	result = torch.argmax(prediction, 1) == Y_test
		# 	accurancy = result.float().mean()
		# 	accContainer.append(accurancy.detach().item())
		# 	print('Accurancy : ', accurancy.item())


	###############################
	# 6. visualize and save results

	x = np.arange(1, sampleNum + 1)
	cost = np.array(costContainer)
	acc = np.array(accContainer)

	f = open("./" + prefix_w + "./costContainer.csv", 'a')
	for n in range(len(costContainer)):
		if n != 0: f.write(",")
		f.write(str(costContainer[n]))
	f.write("\n")
	f.close()

	f = open("./" + prefix_w + "./accContainer.csv", 'a')
	for n in range(len(accContainer)):
		if n != 0: f.write(",")
		f.write(str(accContainer[n]))
	f.write("\n")
	f.close()

	plt.plot(x, cost, color=colorList[-3], label="cost")
	plt.plot(x, acc, color=colorList[-2], label="acc")
	plt.legend()
	plt.grid()
	plt.title("cost, acc")
	plt.savefig(prefix_w + "/testResult.png")
	plt.show()
	plt.clf()

if __name__ == "__main__":
	training_mnist(3333)