import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from module import ParamIO
from module.util import PCA, SUBSPACE
from glob import glob
from tqdm import tqdm

class VGG16_Layerwise1stMag():
    def __init__(self):
        pass

    def __call__(self, csvFolder, layerNum, totalSampleNum):

        pcaTool = PCA.WeightPCA()
        smTool = SUBSPACE.SubspaceDiff()

        if layerNum <= 13:
            result = self._pca_1stMag_layerwise_conv_VGG(pcaTool, smTool, csvFolder, layerNum, totalSampleNum=totalSampleNum)
        else:
            result = self._pca_1stMag_layerwise_fc_VGG(pcaTool, smTool, csvFolder, layerNum, totalSampleNum=totalSampleNum)

        return result

    def _pca_1stMag_layerwise_conv_VGG(self, pcaTool, smTool, csvFolder, layerNum, totalSampleNum=100):

        magContainer = []
        print("layer {}".format(layerNum))
        kernelNumContainer = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        unitNumContainer = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]

        for i in tqdm(range(int(totalSampleNum/3) - 1)):
            _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum)), loadMin=3*i+1, loadMax=3*i+4)
            
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

    def _pca_1stMag_layerwise_fc_VGG(self, pcaTool, smTool, csvFolder, layerNum, totalSampleNum=100):

        magContainer = []
        print("layer {}".format(layerNum))
        coluumnNumContainer = [4096, 4096, 10]
        rowNumContainer = [512, 4096, 4096]
        dimContainer = [512, 4096, 10]

        for i in tqdm(range(int(totalSampleNum/3) - 1)):
            _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum)), loadMin=3*i+1, loadMax=3*i+4)
            
            dataArray1 = np.transpose(np.reshape(weight_layer[:,0], (coluumnNumContainer[layerNum - 1], rowNumContainer[layerNum - 1])))
            dataArray2 = np.transpose(np.reshape(weight_layer[:,1], (coluumnNumContainer[layerNum - 1], rowNumContainer[layerNum - 1])))
            if i == 1:
                print(dataArray1.shape)

            _, basis1 = pcaTool.pca_lowcost(dataArray1, dimContainer[layerNum - 1])
            _, basis2 = pcaTool.pca_lowcost(dataArray2, dimContainer[layerNum - 1])
            basis1 = basis1[:,:dimContainer[layerNum - 1]]
            basis2 = basis2[:,:dimContainer[layerNum - 1]]
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