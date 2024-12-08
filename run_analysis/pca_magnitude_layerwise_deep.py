# ********************************************************************************************
# 파일명 : pca_magnitude_layerwise_deep.py
# 목적　 : 1개의 weight trajectory에 대해 ***step 별로, layer 별로 weight의 부분공간***을 계산,
#          step 별로 각 층의 weight 부분공간이 어떻게 변화하는지 분석하는 모듈
# 구조 　: 함수 구조(CIFAR의 1st, 2nd magnitude, MNIST의 1st, 2nd magnitude)
# ********************************************************************************************

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import module.pcaModule as wp
import module.makeArray as ma
import module.smModule as sm
from glob import glob
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

def pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, layerNum, totalSampleNum=200):

    magContainer = []
    print("layer {}".format(layerNum))
    kernelNumContainer = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    unitNumContainer = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]

    for i in tqdm(range(int(totalSampleNum/2) - 1)):
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum)), loadMin=2*i+1, loadMax=2*i+3)
        
        dataArray1 = np.transpose(np.reshape(weight_layer[:,0], (kernelNumContainer[layerNum - 1], 9*unitNumContainer[layerNum - 1])))
        dataArray2 = np.transpose(np.reshape(weight_layer[:,1], (kernelNumContainer[layerNum - 1], 9*unitNumContainer[layerNum - 1])))
        if i == 1:
            print(dataArray1.shape)

        if layerNum == 1:
            _, basis1 = pcaTool.pca_lowcost(dataArray1, 9*unitNumContainer[layerNum - 1])
            _, basis2 = pcaTool.pca_lowcost(dataArray2, 9*unitNumContainer[layerNum - 1])
            basis1 = basis1[:,:9*unitNumContainer[layerNum - 1]]
            basis2 = basis2[:,:9*unitNumContainer[layerNum - 1]]
        else:
            _, basis1 = pcaTool.pca_lowcost(dataArray1, kernelNumContainer[layerNum - 1])
            _, basis2 = pcaTool.pca_lowcost(dataArray2, kernelNumContainer[layerNum - 1])
            basis1 = basis1[:,:kernelNumContainer[layerNum - 1]]
            basis2 = basis2[:,:kernelNumContainer[layerNum - 1]]
        if i == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
        else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

        # plt.title("first-magnitude of weight subspace(dim={}, layer={})".format(compNumContainer[layerNum - 1], layerNum))
        # plt.xlabel("each step")
        # plt.ylabel("1st magnitude")
        # plt.plot(range(len(magContainer)), magContainer)
        # plt.savefig("result/layer{:02d}_firstmag_{:02d}dim_{:04d}.png".format(layerNum, compNumContainer[layerNum - 1], seed))
        # plt.show()
        # plt.clf()

    return np.array(magContainer)

pcaTool = wp.weightPCA()
smTool = sm.SubspaceDiff()

seed = 3817

csvFolderList = glob("C:/Users/dmtsa/research/run_analysis/DB_CNN_MNIST_1/1-4-8-fc__*")
csvFolder = "C:/Users/dmtsa/research/run_analysis/DB_VGG_CIFAR_1/vgg16__{:04d}".format(seed)

# i1 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 1)
# i2 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 2)
# i3 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 3)
# i4 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 4)
i5 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 5)
i6 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 6)
i7 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 7)
i8 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 8)
i9 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 9)
i10 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 10)
i11 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 11)
i12 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 12)
i13 = pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, seed, csvFolder, 13)

plt.title("first-magnitude of weight subspace(layer : conv 5~13)")
plt.xlabel("each step")
plt.ylabel("1st magnitude")
# plt.plot(range(len(i1)), i1, label="layer conv 1")
# plt.plot(range(len(i2)), i2, label="layer conv 2")
# plt.plot(range(len(i3)), i3, label="layer conv 3")
# plt.plot(range(len(i4)), i4, label="layer conv 4")
plt.plot(range(len(i5)), i5, label="layer conv 5")
plt.plot(range(len(i6)), i6, label="layer conv 6")
plt.plot(range(len(i7)), i7, label="layer conv 7")
plt.plot(range(len(i8)), i8, label="layer conv 8")
plt.plot(range(len(i9)), i9, label="layer conv 9")
plt.plot(range(len(i10)), i10, label="layer conv 10")
plt.plot(range(len(i11)), i11, label="layer conv 11")
plt.plot(range(len(i12)), i12, label="layer conv 12")
plt.plot(range(len(i13)), i13, label="layer conv 13")
plt.grid()
plt.legend()
plt.savefig("result/conv5_to_13_firstmag_VGG_64_to_512dim_{:04d}.png".format(seed))
# plt.show()
plt.clf()