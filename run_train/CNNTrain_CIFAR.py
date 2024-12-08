import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random as rd
import numpy as np
from tqdm import tqdm
from module import netModuleCNN as nmc
import matplotlib.pyplot as plt
import math

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def training_cifar(rdSeed, training_epochs, batch_size, period, model_name, lr=0.005):

	######################################################
	######################################################
	# 0. get random seed and save number from command line

	torch.manual_seed(rdSeed)
	rd.seed(rdSeed)

	###################################################
	###################################################
	# 1-1. set learning parameter and prepare save folder
	learning_rate = lr
	# training_epochs = 100
	# batch_size = 100
	# period = 250

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

	prefix_w = model_name + "__" + str(rdSeed).zfill(4)

	if not os.path.isdir("./" + prefix_w):
		os.mkdir("./" + prefix_w)

	###########################################################
	###########################################################
	# 2. import CIFAR dataset as training data and testing data

	transform = transforms.Compose(
    [
	 #transforms.Resize(224),
	 #transforms.CenterCrop(224),
	 transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='C:/Users/dmtsa/RESEARCH/DB', train=True,
											download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='C:/Users/dmtsa/RESEARCH/DB', train=False,
										download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat',
			'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


	###########################################################
	###########################################################
	# 3. select model, loss function and optimizer for training
	if model_name == "resnet18":
		model = nmc.RESNET18_CIFAR_small(10)
	elif model_name == "VGG16":
		model = nmc.VGG16(64)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

	a = 1
	for name, param in model.named_parameters():
		if "bias" in name: continue
		print("{:02d} : ".format(a))
		print(name,param.shape)
		a += 1
	paramNum = 0
	for param in model.parameters():
		paramNum = paramNum + 1

	#########################################
	# 4. process raw data into pytorch tensor
	# X_test = torch.stack(test_data, dim=0)
	# Y_test = torch.stack(test_label, dim=0)
	# Y_test = Y_test.squeeze()
	# Y_test = torch.argmax(Y_test, 1)

	flat = nn.Flatten(start_dim=0)


	##########################
	# 5. execute training step
	costContainer = []
	accContainer = []

	for epoch in tqdm(range(training_epochs)):

		model.train()

		for i, data in enumerate(trainloader, 0):

			X, Y = data

			optimizer.zero_grad()
			prediction = model(X)
			cost = criterion(prediction, Y)

			cost.backward()
			optimizer.step()

			if i%(period) == 0:

				
				counter = 0
				for name, param in model.named_parameters(): # iteration : each parameter
					if "bias" in name: continue
					param_flatten = flat(param)
					param_flatten = param_flatten.detach().tolist()
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
				if i == (len(trainloader) - period):
					print()
					print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, cost.detach().item()), end=" || ")

				with torch.no_grad():
					model.eval()
					correct = 0
					total = 0
					total_test_loss = 0
					cnt = 0
					for data in testloader:
						images, labels = data
						outputs = model(images)

						test_loss = criterion(outputs, labels)
						total_test_loss += test_loss
						cnt += 1

						_, predicted = torch.max(outputs, 1)
						c = (predicted == labels).squeeze()
						for j in range(batch_size):
							correct += c[j].item()
							total += 1
					total_test_loss = total_test_loss / cnt
					scheduler.step(total_test_loss)
					accContainer.append(correct / total)
					if i == (len(trainloader) - period): print('Accuracy = {:>.9}%'.format(100 * correct / total))

		


	###############################
	# 6. visualize and save results

	# paramSpace = np.arange(1, sampleNum + 1)
	# testSpace = np.arange(1, training_epochs + 1) * (sampleNum / training_epochs)

	# cost = np.array(costContainer) * costMultiplier
	# acc = np.array(accContainer) * accMultiplier

	# for i in range(paramNum):
	# 	plt.plot(paramSpace, paramNormContainer[i], color=colorList[i], label="norm of layer{}".format(i))
	# plt.plot(testSpace, cost, color=colorList[-3], label="cost[X{}]".format(costMultiplier))
	# plt.plot(testSpace, acc, color=colorList[-2], label="acc[X{}]".format(accMultiplier))
	# plt.legend()
	# plt.grid()
	# plt.title("cost, acc, and parameter norm")
	# plt.savefig(prefix_w + "/testResult_paramNorm.png")
	# plt.show()
	# plt.clf()

	# for i in range(paramNum):
	# 	plt.plot(paramSpace, gradNormContainer[i], color=colorList[i], label="gradient norm of layer{}".format(i))
	# plt.plot(testSpace, cost, color=colorList[-3], label="cost[X{}]".format(costMultiplier))
	# plt.plot(testSpace, acc, color=colorList[-2], label="acc[X{}]".format(accMultiplier))
	# plt.legend()
	# plt.grid()
	# plt.title("cost, acc, and gradient norm")
	# plt.savefig(prefix_w + "/testResult_gradNorm.png")
	# plt.show()
	# plt.clf()

	x = np.arange(1, len(costContainer) + 1)
	cost = np.array(costContainer)
	acc = np.array(accContainer) * 5

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
	plt.plot(x, acc, color=colorList[-2], label="acc[x 5]")
	plt.legend()
	plt.grid()
	plt.title("cost, acc")
	plt.savefig(prefix_w + "/testResult.png")
	# plt.show()
	plt.clf()

if __name__ == "__main__":
	training_cifar(7982, 100, 100, 250, "VGG16", lr=0.01)