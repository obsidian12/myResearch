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
from module import netModuleFC as nmf
from datetime import datetime
import matplotlib.pyplot as plt
import math
import module.pcaModule as wp
from glob import glob

def training_MNIST(saveNum, rdSeed, learning_rate, training_epochs, batch_size, test_size, item_num, sampling_num, d,
			 totalDataNum, train_data, test_data, train_label, test_label, basis, initFlag, frozenFlag, colorList, accMultiplier, costMultiplier, gradNormMultiplier,
			 frozenBuffer):
	
	###################################################
	###################################################
	# 1-1. set random seeds
	torch.manual_seed(rdSeed)
	rd.seed(rdSeed)


	###################################################
	###################################################
	# 1-2. prepare save folder
	runDate = datetime.today().strftime("%Y_%m_%d")
	prefix_tr = runDate + "_" +  str(saveNum).zfill(3) + "_lda_tr"
	prefix_sd = runDate + "_" + str(saveNum).zfill(3) + "_lda_sd"

	if not os.path.isdir("./" + prefix_sd):
			os.mkdir("./" + prefix_sd)
	if not os.path.isdir("./" + prefix_tr):
			os.mkdir("./" + prefix_tr)
	

	###########################################################
	###########################################################
	# 3. select model, loss function and optimizer for training
	model = nmc.CNN_MNIST_LDA2(item_num, d)
	model.import_basis(basis[:,:d], initFlag)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	counter = 0
	param_num_container = []
	for param in model.parameters():
		if param.requires_grad:
			print("param found : layer {}".format(counter))
			param_num_container.append(counter)
		counter = counter + 1
	param_num = len(param_num_container)

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
	period = int(iterNum * (training_epochs / sampling_num))

	frozenFlagContainer = []
	for i in range(d):
		frozenFlagContainer.append(False)

	for i in range(param_num):
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

			if frozenFlag and epoch * iterNum + i > 20:

				for paramCounter, param in enumerate(model.parameters()):
					if paramCounter == 0:
						tmp = []
						for paramDim in range(d):
							tmp.append(abs(param.grad[paramDim]))
						if frozenBuffer.insert(tmp, epoch * iterNum + i):
							frozenFlagContainer = frozenBuffer.getFlagContainer()
						for paramDim in range(d):
							if frozenFlagContainer[paramDim]: param.grad[paramDim] = 0

			if i%(period) == 0:
				counter = 0
				for param in model.parameters():
					if param.grad is not None:
						tmpGrad = flat(param.grad.detach()).tolist()
						gradNormContainer[counter].append(np.linalg.norm(np.array(tmpGrad))) # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
						counter = counter + 1
						
						fp = open("./" + prefix_sd + "/layer{}_grad.csv".format(counter), 'a')
						for j in range(len(tmpGrad)):
							if j != 0: fp.write(",")
							fp.write(str(tmpGrad[j]))
						fp.write("\n")
						fp.close()

			optimizer.step()

			avg_cost = avg_cost + cost / iterNum

			if i%(period) == 0:

				counter = 0
				for idx, param in enumerate(model.parameters()):
					if idx in param_num_container:
						tmp = flat(param)
						tmp = tmp.detach().tolist()
						paramNormContainer[counter].append(np.linalg.norm(np.array(tmp))) # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
						counter = counter + 1

						# save each parameter elements in divided csv file
						partitionNum = math.ceil(len(tmp) / 1000)
						for m in range(partitionNum):
							if not os.path.isdir("./" + prefix_sd + "/layer{}".format(counter)):
								os.mkdir("./" + prefix_sd + "/layer{}".format(counter))
							f = open("./" + prefix_sd + "/layer{}/part{:04d}.csv".format(counter, m), 'a')
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
			print('Accurancy : {}%({})'.format(round(accurancy.item() * test_size), accurancy.item()))

			if (epoch + 1 == training_epochs) or (epoch + 1 == int(training_epochs / 2)):
				f = open("./CNN_MNIST_LDA_test.csv", "a")
				f.write(str(saveNum) + ",")
				f.write(str(rdSeed) + ",")
				f.write(str(learning_rate) + ",")
				f.write(str(epoch + 1) + ",")
				f.write(str(d) + ",")
				f.write(str(frozenFlag) + ",")
				f.write(str(initFlag) + ",")
				f.write(str(accurancy.item()))
				f.write("\n")
				f.close()

				f = open("./CNN_ALOI_LDA_frozenContainer.csv", "a")
				f.write(str(frozenFlagContainer)[1:-1])
				f.write("\n")
				f.close()


	###############################
	# 6. visualize and save results
	model.save_models(prefix_tr + "/final")

	paramSpace = np.arange(1, sampling_num + 1)
	testSpace = np.arange(1, training_epochs + 1) * (sampling_num / training_epochs)

	cost = np.array(costContainer) * costMultiplier
	acc = np.array(accContainer) * accMultiplier

	for i in range(param_num):
		plt.plot(paramSpace, paramNormContainer[i], color=colorList[i], label="norm of layer{}".format(i))
	plt.plot(testSpace, cost, color=colorList[-3], label="cost[X{}]".format(costMultiplier))
	plt.plot(testSpace, acc, color=colorList[-2], label="acc[X{}]".format(accMultiplier))
	plt.legend()
	plt.grid()
	plt.title("cost, acc, and parameter norm")
	plt.savefig(prefix_sd + "/testResult_paramNorm.png")
	#plt.show()
	plt.clf()

	for i in range(param_num):
		plt.plot(paramSpace, gradNormContainer[i], color=colorList[i], label="gradient norm of layer{}".format(i))
	plt.plot(testSpace, cost, color=colorList[-3], label="cost[X{}]".format(costMultiplier))
	plt.plot(testSpace, acc, color=colorList[-2], label="acc[X{}]".format(accMultiplier))
	plt.legend()
	plt.grid()
	plt.title("cost, acc, and gradient norm")
	plt.savefig(prefix_sd + "/testResult_gradNorm.png")
	#plt.show()
	plt.clf()

def training_MNISTFC(saveNum, rdSeed, learning_rate, training_epochs, batch_size, test_size, item_num, sampling_num, d,
			 totalDataNum, train_data, test_data, train_label, test_label, basis, initFlag, frozenFlag, colorList, accMultiplier, costMultiplier, gradNormMultiplier,
			 frozenBuffer):
	
	###################################################
	###################################################
	# 1-1. set random seeds
	torch.manual_seed(rdSeed)
	rd.seed(rdSeed)


	###################################################
	###################################################
	# 1-2. prepare save folder
	runDate = datetime.today().strftime("%Y_%m_%d")
	prefix_tr = runDate + "_" +  str(saveNum).zfill(3) + "_lda_tr"
	prefix_sd = runDate + "_" + str(saveNum).zfill(3) + "_lda_sd"

	if not os.path.isdir("./" + prefix_sd):
			os.mkdir("./" + prefix_sd)
	if not os.path.isdir("./" + prefix_tr):
			os.mkdir("./" + prefix_tr)
	

	###########################################################
	###########################################################
	# 3. select model, loss function and optimizer for training
	model = nmf.FC_MNIST_LDA2(item_num, d)
	model.import_basis(basis[:,:d], initFlag)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	counter = 0
	param_num_container = []
	for param in model.parameters():
		if param.requires_grad:
			print("param found : layer {}".format(counter))
			param_num_container.append(counter)
		counter = counter + 1
	param_num = len(param_num_container)

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
	period = int(iterNum * (training_epochs / sampling_num))

	frozenFlagContainer = []
	for i in range(d):
		frozenFlagContainer.append(False)

	for i in range(param_num):
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

			if frozenFlag and epoch * iterNum + i > 20:

				for paramCounter, param in enumerate(model.parameters()):
					if paramCounter == 0:
						tmp = []
						for paramDim in range(d):
							tmp.append(abs(param.grad[paramDim]))
						if frozenBuffer.insert(tmp, epoch * iterNum + i):
							frozenFlagContainer = frozenBuffer.getFlagContainer()
						for paramDim in range(d):
							if frozenFlagContainer[paramDim]: param.grad[paramDim] = 0

			if i%(period) == 0:
				counter = 0
				for param in model.parameters():
					if param.grad is not None:
						tmpGrad = flat(param.grad.detach()).tolist()
						gradNormContainer[counter].append(np.linalg.norm(np.array(tmpGrad))) # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
						counter = counter + 1
						
						fp = open("./" + prefix_sd + "/layer{}_grad.csv".format(counter), 'a')
						for j in range(len(tmpGrad)):
							if j != 0: fp.write(",")
							fp.write(str(tmpGrad[j]))
						fp.write("\n")
						fp.close()

			optimizer.step()

			avg_cost = avg_cost + cost / iterNum

			if i%(period) == 0:

				counter = 0
				for idx, param in enumerate(model.parameters()):
					if idx in param_num_container:
						tmp = flat(param)
						tmp = tmp.detach().tolist()
						paramNormContainer[counter].append(np.linalg.norm(np.array(tmp))) # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
						counter = counter + 1

						# save each parameter elements in divided csv file
						partitionNum = math.ceil(len(tmp) / 1000)
						for m in range(partitionNum):
							if not os.path.isdir("./" + prefix_sd + "/layer{}".format(counter)):
								os.mkdir("./" + prefix_sd + "/layer{}".format(counter))
							f = open("./" + prefix_sd + "/layer{}/part{:04d}.csv".format(counter, m), 'a')
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
			print('Accurancy : {}%({})'.format(round(accurancy.item() * test_size), accurancy.item()))

			if (epoch + 1 == training_epochs) or (epoch + 1 == int(training_epochs / 2)):
				f = open("./CNN_MNIST_LDA_test.csv", "a")
				f.write(str(saveNum) + ",")
				f.write(str(rdSeed) + ",")
				f.write(str(learning_rate) + ",")
				f.write(str(epoch + 1) + ",")
				f.write(str(d) + ",")
				f.write(str(frozenFlag) + ",")
				f.write(str(initFlag) + ",")
				f.write(str(accurancy.item()))
				f.write("\n")
				f.close()

				f = open("./CNN_ALOI_LDA_frozenContainer.csv", "a")
				f.write(str(frozenFlagContainer)[1:-1])
				f.write("\n")
				f.close()


	###############################
	# 6. visualize and save results
	model.save_models(prefix_tr + "/final")

	paramSpace = np.arange(1, sampling_num + 1)
	testSpace = np.arange(1, training_epochs + 1) * (sampling_num / training_epochs)

	cost = np.array(costContainer) * costMultiplier
	acc = np.array(accContainer) * accMultiplier

	for i in range(param_num):
		plt.plot(paramSpace, paramNormContainer[i], color=colorList[i], label="norm of layer{}".format(i))
	plt.plot(testSpace, cost, color=colorList[-3], label="cost[X{}]".format(costMultiplier))
	plt.plot(testSpace, acc, color=colorList[-2], label="acc[X{}]".format(accMultiplier))
	plt.legend()
	plt.grid()
	plt.title("cost, acc, and parameter norm")
	plt.savefig(prefix_sd + "/testResult_paramNorm.png")
	#plt.show()
	plt.clf()

	for i in range(param_num):
		plt.plot(paramSpace, gradNormContainer[i], color=colorList[i], label="gradient norm of layer{}".format(i))
	plt.plot(testSpace, cost, color=colorList[-3], label="cost[X{}]".format(costMultiplier))
	plt.plot(testSpace, acc, color=colorList[-2], label="acc[X{}]".format(accMultiplier))
	plt.legend()
	plt.grid()
	plt.title("cost, acc, and gradient norm")
	plt.savefig(prefix_sd + "/testResult_gradNorm.png")
	#plt.show()
	plt.clf()
	
def training_ALOI(saveNum, rdSeed, learning_rate, training_epochs, batch_size, test_size, item_num, sampling_num, d,
			 train_data, test_data, label, basis, initFlag, frozenFlag, colorList, accMultiplier, costMultiplier, gradNormMultiplier,
			 frozenBuffer):

	###################################################
	###################################################
	# 1-1. set random seeds
	torch.manual_seed(rdSeed)
	rd.seed(rdSeed)


	###################################################
	###################################################
	# 1-2. prepare save folder
	runDate = datetime.today().strftime("%Y_%m_%d")
	prefix_tr = runDate + "_" +  str(saveNum).zfill(3) + "_lda_tr"
	prefix_sd = runDate + "_" + str(saveNum).zfill(3) + "_lda_sd"

	if not os.path.isdir("./" + prefix_sd):
			os.mkdir("./" + prefix_sd)
	if not os.path.isdir("./" + prefix_tr):
			os.mkdir("./" + prefix_tr)


	###########################################################
	###########################################################
	# 4. select model, loss function and optimizer for training
	model = nmc.CNN_ALOI_LDA2(item_num, d)
	model.import_basis(basis[:,:d], initFlag)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	counter = 0
	param_num_container = []
	for param in model.parameters():
		if param.requires_grad:
			print("param found : layer {}".format(counter))
			param_num_container.append(counter)
		counter = counter + 1
	param_num = len(param_num_container)


	#########################################
	#########################################
	# 5. process raw data into pytorch tensor
	X_test_raw, Y_test_raw = um.sample(test_data, label, test_size, 36 * item_num)
	X_test = torch.stack(X_test_raw, dim=0)
	Y_test = torch.stack(Y_test_raw, dim=0)
	Y_test = Y_test.squeeze()
	Y_test = torch.argmax(Y_test, 1)

	flat = nn.Flatten(start_dim=0)


	##########################
	##########################
	# 5. execute training step
	costContainer = []
	accContainer = []
	paramNormContainer = [] # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
	gradNormContainer = []
	for i in range(param_num):
		paramNormContainer.append([])
		gradNormContainer.append([])

	sampler = um.sampler()
	totaldata_num = 36 * item_num
	iteration_num = int(totaldata_num / batch_size)
	period = int(iteration_num * (training_epochs / sampling_num))

	frozenFlagContainer = []
	for i in range(d):
		frozenFlagContainer.append(False)

	for epoch in tqdm(range(training_epochs)):
		avg_cost = 0
		sampler.setBuffer(totaldata_num)

		for i in range(iteration_num):

			x_raw, y_raw = sampler.sample(train_data, label, batch_size)

			X = torch.stack(x_raw, dim=0)
			Y = torch.stack(y_raw, dim=0)
			Y = Y.squeeze()

			optimizer.zero_grad()
			prediction = model(X)
			cost = criterion(prediction, Y)

			cost.backward()

			if frozenFlag and epoch * iteration_num + i > 20:

				# if epoch * (sampling_num / training_epochs) + i // period >= frozenStep:
				# 	for paramCounter, param in enumerate(model.parameters()):
				# 		if paramCounter == 0:
				# 			for frozenDim in frozenDimList:
				# 				param.grad[frozenDim] = 0

				for paramCounter, param in enumerate(model.parameters()):
					if paramCounter == 0:
						tmp = []
						for paramDim in range(d):
							tmp.append(abs(param.grad[paramDim]))
						if frozenBuffer.insert(tmp, epoch * iteration_num + i):
							frozenFlagContainer = frozenBuffer.getFlagContainer()
						for paramDim in range(d):
							if frozenFlagContainer[paramDim]: param.grad[paramDim] = 0
			
			# save norm of gradient of every layer as list in every (period) steps
			# save gradient vector of every layer as csv file in every (period) steps
			if i%(period) == 0:
				counter = 0
				for param in model.parameters():
					if param.grad is not None:
						tmpGrad = flat(param.grad.detach()).tolist()
						gradNormContainer[counter].append(np.linalg.norm(np.array(tmpGrad)) * gradNormMultiplier) # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
						counter = counter + 1

						fp = open("./" + prefix_sd + "/layer{}_grad.csv".format(counter), 'a')
						for j in range(len(tmpGrad)):
							if j != 0: fp.write(",")
							fp.write(str(tmpGrad[j]))
						fp.write("\n")
						fp.close()

			optimizer.step()

			avg_cost = avg_cost + cost / iteration_num
			
			# save norm of parameter of every layer as list in every (period) steps
			# save parameter vector of every layer in every (period) steps
			if i%(period) == 0:
				counter = 0
				for idx, param in enumerate(model.parameters()):
					if idx in param_num_container:
						tmp = flat(param)
						tmp = tmp.detach().tolist()
						paramNormContainer[counter].append(np.linalg.norm(np.array(tmp))) # 차원이 너무 크므로 그냥 즉석에서 노름만 계산해서 리스트에 저장
						counter = counter + 1

						# save each parameter elements in divided csv file
						partitionNum = math.ceil(len(tmp) / 1000)
						for m in range(partitionNum):
							if not os.path.isdir("./" + prefix_sd + "/layer{}".format(counter)):
								os.mkdir("./" + prefix_sd + "/layer{}".format(counter))
							f = open("./" + prefix_sd + "/layer{}/part{:04d}.csv".format(counter, m), 'a')
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
			print('Accurancy : {}%({})'.format(round(accurancy.item() * test_size), accurancy.item()))
			
			if (epoch + 1 == training_epochs) or (epoch + 1 == int(training_epochs / 2)):
				f = open("./CNN_ALOI_LDA_test.csv", "a")
				f.write(str(saveNum) + ",")
				f.write(str(rdSeed) + ",")
				f.write(str(learning_rate) + ",")
				f.write(str(epoch + 1) + ",")
				f.write(str(d) + ",")
				f.write(str(frozenFlag) + ",")
				f.write(str(initFlag) + ",")
				f.write(str(accurancy.item()))
				f.write("\n")
				f.close()

				f = open("./CNN_ALOI_LDA_frozenContainer.csv", "a")
				f.write(str(frozenFlagContainer)[1:-1])
				f.write("\n")
				f.close()


	###############################
	###############################
	# 6. visualize and save results
	model.save_models(prefix_tr + "/final")

	paramSpace = np.arange(1, sampling_num + 1)
	testSpace = np.arange(1, training_epochs + 1) * (sampling_num / training_epochs)

	# cost = np.array(costContainer) * costMultiplier
	acc = np.array(accContainer) * accMultiplier

	for i in range(param_num):
		plt.plot(paramSpace, paramNormContainer[i], color=colorList[i], label="norm of layer{}".format(param_num_container[i] + 1))
	# plt.plot(testSpace, cost, color=colorList[-3], label="cost[X{}]".format(costMultiplier))
	plt.plot(testSpace, acc, color=colorList[-2], label="acc[X{}]".format(accMultiplier))
	plt.legend()
	plt.grid()
	plt.title("cost, acc, and parameter norm")
	plt.savefig(prefix_sd + "/testResult_paramNorm.png")
	#plt.show()
	plt.clf()

	for i in range(param_num):
		plt.plot(paramSpace, gradNormContainer[i], color=colorList[i], label="gradient norm of layer{}[X{}]".format(param_num_container[i] + 1, gradNormMultiplier))
	# plt.plot(testSpace, cost, color=colorList[-3], label="cost[X{}]".format(costMultiplier))
	plt.plot(testSpace, acc, color=colorList[-2], label="acc[X{}]".format(accMultiplier))
	plt.legend()
	plt.grid()
	plt.title("cost, acc, and gradient norm")
	plt.savefig(prefix_sd + "/testResult_gradNorm.png")
	#plt.show()
	plt.clf()