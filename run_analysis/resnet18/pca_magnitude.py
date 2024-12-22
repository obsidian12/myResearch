import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import numpy as np
from module import ParamIO
from module.util import PCA, SUBSPACE
from glob import glob
from tqdm import tqdm

class Resnet18_IntegratedMag():

    def __init__(self):
        pass

    def intagrated_1stMag(pcaTool, smTool, csvFolder, n_components):

        magContainer = []

        for i in tqdm(range(1, 1000)):
            _, weight_layer01 = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer1/part*.csv"), [i, i + 2])
            _, weight_layer02 = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer2/part*.csv") + glob(csvFolder + "/layer3/part*.csv") + glob(csvFolder + "/layer4/part*.csv"), [i, i + 2])
            _, weight_layer03 = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer5/part*.csv") + glob(csvFolder + "/layer6/part*.csv"), i, i + 2)
            _, weight_layer04 = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer7/part*.csv") + glob(csvFolder + "/layer8/part*.csv") + glob(csvFolder + "/layer9/part*.csv"), [i, i + 2])
            _, weight_layer05 = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer10/part*.csv") + glob(csvFolder + "/layer11/part*.csv"), i, i + 2)
            _, weight_layer06 = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer12/part*.csv") + glob(csvFolder + "/layer13/part*.csv") + glob(csvFolder + "/layer14/part*.csv"), [i, i + 2])
            _, weight_layer07 = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer15/part*.csv") + glob(csvFolder + "/layer16/part*.csv"), i, i + 2)
            _, weight_layer08 = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer17/part*.csv") + glob(csvFolder + "/layer18/part*.csv") + glob(csvFolder + "/layer19/part*.csv"), [i, i + 2])
            _, weight_layer09 = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer20/part*.csv") + glob(csvFolder + "/layer21/part*.csv"), i, i + 2)
            _, weight_layer10 = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer22/part*.csv") + glob(csvFolder + "/layer23/part*.csv"), i, i + 2)

            dataArray11 = np.pad(weight_layer01[:,0], (0,114), 'constant', constant_values=0)[:, None]
            dataArray12 = np.pad(np.transpose(np.reshape(weight_layer02[0:3456,0], (3, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray12[1152:1280,0] = weight_layer02[3456:,0]
            dataArray13 = np.pad(np.transpose(np.reshape(weight_layer03[:,0], (4, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray14 = np.pad(np.transpose(np.reshape(weight_layer04[0:13824,0], (12, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray14[1152:1280,0:4] = np.transpose(np.reshape(weight_layer04[13824:,0], (4, 128)))
            dataArray15 = np.pad(np.transpose(np.reshape(weight_layer05[:,0], (16, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray16 = np.pad(np.transpose(np.reshape(weight_layer06[0:55296,0], (48, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray16[1152:1280,0:16] = np.transpose(np.reshape(weight_layer06[55296:,0], (16, 128)))
            dataArray17 = np.pad(np.transpose(np.reshape(weight_layer07[:,0], (64, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray18 = np.pad(np.transpose(np.reshape(weight_layer08[0:221184,0], (192, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray18[1152:1280,0:64] = np.transpose(np.reshape(weight_layer08[221184:,0], (64, 128)))
            dataArray19 = np.pad(np.transpose(np.reshape(weight_layer09[:,0], (256, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray20 = weight_layer10[:,0][:, None]

            dataArray1 = np.concatenate((dataArray11, dataArray12, dataArray13, dataArray14, dataArray15, dataArray16, dataArray17, dataArray18, dataArray19, dataArray20), axis=1)

            dataArray21 = np.pad(weight_layer01[:,1], (0,114), 'constant', constant_values=0)[:, None]
            dataArray22 = np.pad(np.transpose(np.reshape(weight_layer02[0:3456,1], (3, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray22[1152:1280,0] = weight_layer02[3456:,1]
            dataArray23 = np.pad(np.transpose(np.reshape(weight_layer03[:,1], (4, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray24 = np.pad(np.transpose(np.reshape(weight_layer04[0:13824,1], (12, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray24[1152:1280,0:4] = np.transpose(np.reshape(weight_layer04[13824:,1], (4, 128)))
            dataArray25 = np.pad(np.transpose(np.reshape(weight_layer05[:,1], (16, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray26 = np.pad(np.transpose(np.reshape(weight_layer06[0:55296,1], (48, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray26[1152:1280,0:16] = np.transpose(np.reshape(weight_layer06[55296:,1], (16, 128)))
            dataArray27 = np.pad(np.transpose(np.reshape(weight_layer07[:,1], (64, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray28 = np.pad(np.transpose(np.reshape(weight_layer08[0:221184,1], (192, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray28[1152:1280,0:64] = np.transpose(np.reshape(weight_layer08[221184:,1], (64, 128)))
            dataArray29 = np.pad(np.transpose(np.reshape(weight_layer09[:,1], (256, 1152))), ((0,138),(0,0)), 'constant', constant_values=0)
            dataArray30 = weight_layer10[:,1][:, None]

            dataArray2 = np.concatenate((dataArray21, dataArray22, dataArray23, dataArray24, dataArray25, dataArray26, dataArray27, dataArray28, dataArray29, dataArray30), axis=1)

            _, tmp1 = pcaTool.pca_basic(dataArray1)
            basis1 = tmp1[:,0:n_components]
            _, tmp2 = pcaTool.pca_basic(dataArray2)
            basis2 = tmp2[:,0:n_components]
            magContainer.append(smTool.calc_magnitude(basis1, basis2))

        return np.array(magContainer)
    
class Resnet18_Layerwise1stMag():

    def __init__(self):
        pass

    def __call__(self, csvFolder, layerNum, sublayerNum, totalSampleNum):

        pcaTool = PCA.WeightPCA()
        smTool = SUBSPACE.SubspaceDiff()
        
        if layerNum == 1:
            result = self._pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, csvFolder, totalSampleNum=totalSampleNum)
        elif layerNum == 6:
            result = self._pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, csvFolder, totalSampleNum=totalSampleNum)
        else:
            if sublayerNum == 1:
                result = self._pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, csvFolder, layerNum-1, totalSampleNum=totalSampleNum)
            else:
                result = self._pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, csvFolder, layerNum-1, sublayerNum-1, totalSampleNum=totalSampleNum)

        return result

    def _pca_1stMag_layerwise_ip_CIFAR(self, pcaTool, smTool, csvFolder, totalSampleNum=100):

        magContainer = []
        print("layer input")

        for i in tqdm(range(totalSampleNum-1)):
            _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer1/part*.csv"), [i+1, i+2])

            dataArray1 = np.transpose(np.reshape(weight_layer[:,0], (8, 147)))
            dataArray2 = np.transpose(np.reshape(weight_layer[:,1], (8, 147)))
            if i == 1:
                print(dataArray1.shape)

            _, basis1 = pcaTool.pca_lowcost(dataArray1, 8)
            _, basis2 = pcaTool.pca_lowcost(dataArray2, 8)
            basis1 = basis1[:,:8]
            basis2 = basis2[:,:8]
            if i == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

            if i == totalSampleNum-1 - 1 : print(smTool.calc_magnitude(basis1, basis2))

        return np.array(magContainer)
    
    def _pca_1stMag_layerwise_bbi_CIFAR(self, pcaTool, smTool, csvFolder, layerNum, totalSampleNum=100):

        magContainer = []
        print("layer {}(bb(input))".format(layerNum))

        for i in tqdm(range(totalSampleNum-1)):
            _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 3)), [i+1, i+2])

            dataArray1 = np.transpose(np.reshape(weight_layer[:,0], (8*(2**layerNum), 36*(2**layerNum))))
            dataArray2 = np.transpose(np.reshape(weight_layer[:,1], (8*(2**layerNum), 36*(2**layerNum))))
            if i == 1:
                print(dataArray1.shape)

            _, basis1 = pcaTool.pca_lowcost(dataArray1, 8*(2**layerNum))
            _, basis2 = pcaTool.pca_lowcost(dataArray2, 8*(2**layerNum))
            basis1 = basis1[:,:8*(2**layerNum)]
            basis2 = basis2[:,:8*(2**layerNum)]
            if i == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

            if i == totalSampleNum-1 - 1 : print(smTool.calc_magnitude(basis1, basis2))

        return np.array(magContainer)
    
    def _pca_1stMag_layerwise_bb_CIFAR(self, pcaTool, smTool, csvFolder, layerNum, sublayerNum, totalSampleNum=100):

        magContainer = []
        print("layer {}(bb({}th))".format(layerNum, sublayerNum))

        for i in tqdm(range(totalSampleNum-1)):
            if sublayerNum == 1:
                _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), [i+1, i+2])
            elif sublayerNum == 2:
                _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)), [i+1, i+2])
            elif sublayerNum == 3:
                _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), [i+1, i+2])
            
            dataArray1 = np.transpose(np.reshape(weight_layer[:,0], (8*(2**layerNum), 72*(2**layerNum))))
            dataArray2 = np.transpose(np.reshape(weight_layer[:,1], (8*(2**layerNum), 72*(2**layerNum))))
            if i == 1:
                print(dataArray1.shape)

            _, basis1 = pcaTool.pca_lowcost(dataArray1, 8*(2**layerNum))
            _, basis2 = pcaTool.pca_lowcost(dataArray2, 8*(2**layerNum))
            basis1 = basis1[:,:8*(2**layerNum)]
            basis2 = basis2[:,:8*(2**layerNum)]
            if i == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

            if i == totalSampleNum-1 - 1 : print(smTool.calc_magnitude(basis1, basis2))

        return np.array(magContainer)
    
    def _pca_1stMag_layerwise_fc_CIFAR(self, pcaTool, smTool, csvFolder, totalSampleNum=100):

        magContainer = []
        print("layer fc")

        for i in tqdm(range(totalSampleNum-1)):
            _, weight_layerw = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer22/part*.csv"), [i+1, i+2])

            dataArray1w = np.transpose(np.reshape(weight_layerw[:,0], (10, 128)))
            if i == 1:
                print(dataArray1w.shape)

            dataArray2w = np.transpose(np.reshape(weight_layerw[:,1], (10, 128)))

            _, basis1 = pcaTool.pca_lowcost(dataArray1w, 10)
            _, basis2 = pcaTool.pca_lowcost(dataArray2w, 10)
            basis1 = basis1[:,:10]
            basis2 = basis2[:,:10]
            if i == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

            if i == totalSampleNum-1 - 1 : print(smTool.calc_magnitude(basis1, basis2))

        return np.array(magContainer)

class Resnet18_Layerwise2ndMag():
    
    def __init__(self):
        pass

    def __call__(self, csvFolder, layerNum, sublayerNum, totalSampleNum):

        pcaTool = PCA.WeightPCA()
        smTool = SUBSPACE.SubspaceDiff()

        if layerNum == 1:
            result = self._pca_2ndMag_layerwise_ip_CIFAR(pcaTool, smTool, csvFolder, totalSampleNum=totalSampleNum)
        elif layerNum == 6:
            result = self._pca_2ndMag_layerwise_fc_CIFAR(pcaTool, smTool, csvFolder, totalSampleNum=totalSampleNum)
        else:
            if sublayerNum == 1:
                result = self._pca_2ndMag_layerwise_bbi_CIFAR(pcaTool, smTool, csvFolder, layerNum-1, totalSampleNum=totalSampleNum)
            else:
                result = self._pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, csvFolder, layerNum-1, sublayerNum-1, totalSampleNum=totalSampleNum)

        return result
    
    def _pca_2ndMag_layerwise_ip_CIFAR(self, pcaTool, smTool, csvFolder, totalSampleNum=100):

        magContainer = []
        print("layer input")

        for i in tqdm(range(totalSampleNum-1 - 1)):
            _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer1/part*.csv"), [i + 1, (i + 1) + 1, (i + 2)])

            dataArray1 = np.transpose(np.reshape(weight_layer[:,0], (8, 147)))
            dataArray2 = np.transpose(np.reshape(weight_layer[:,1], (8, 147)))
            dataArray3 = np.transpose(np.reshape(weight_layer[:,2], (8, 147)))
            if i == 1:
                print(dataArray1.shape)

            _, basis1 = pcaTool.pca_lowcost(dataArray1, 8)
            _, basis2 = pcaTool.pca_lowcost(dataArray2, 8)
            _, basis3 = pcaTool.pca_lowcost(dataArray3, 8)
            basis1 = basis1[:,:8]
            basis2 = basis2[:,:8]
            basis2 = basis2[:,:8]
            _, k = smTool.calc_karcher_subspace(basis1, basis3, 8)

            if i == 0: magContainer.append(smTool.calc_magnitude(basis2, k, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis2, k))

        return np.array(magContainer)
    
    def _pca_2ndMag_layerwise_bbi_CIFAR(self, pcaTool, smTool, csvFolder, layerNum, totalSampleNum=100):

        magContainer = []
        print("layer {}(bb(input))".format(layerNum))

        for i in tqdm(range(totalSampleNum-1 - 1)):
            _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 3)), [i + 1, (i + 1) + 1, (i + 2)])

            dataArray1 = np.transpose(np.reshape(weight_layer[:,0], (8*(2**layerNum), 36*(2**layerNum))))
            dataArray2 = np.transpose(np.reshape(weight_layer[:,1], (8*(2**layerNum), 36*(2**layerNum))))
            dataArray3 = np.transpose(np.reshape(weight_layer[:,2], (8*(2**layerNum), 36*(2**layerNum))))
            if i == 1:
                print(dataArray1.shape)

            _, basis1 = pcaTool.pca_lowcost(dataArray1, 8*(2**layerNum))
            _, basis2 = pcaTool.pca_lowcost(dataArray2, 8*(2**layerNum))
            _, basis3 = pcaTool.pca_lowcost(dataArray3, 8*(2**layerNum))
            basis1 = basis1[:,:8*(2**layerNum)]
            basis2 = basis2[:,:8*(2**layerNum)]
            basis3 = basis3[:,:8*(2**layerNum)]
            _, k = smTool.calc_karcher_subspace(basis1, basis3, 8*(2**layerNum))

            if i == 0: magContainer.append(smTool.calc_magnitude(basis2, k, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis2, k))

        return np.array(magContainer)
    
    def _pca_2ndMag_layerwise_bb_CIFAR(self, pcaTool, smTool, csvFolder, layerNum, sublayerNum, totalSampleNum=100):

        magContainer = []
        print("layer {}(bb({}th))".format(layerNum, sublayerNum))

        for i in tqdm(range(totalSampleNum-1 - 1)):
            if sublayerNum == 1:
                _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), [i + 1, (i + 1) + 1, (i + 2)])
            elif sublayerNum == 2:
                _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)), [i + 1, (i + 1) + 1, (i + 2)])
            elif sublayerNum == 3:
                _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), [i + 1, (i + 1) + 1, (i + 2)])
            
            dataArray1 = np.transpose(np.reshape(weight_layer[:,0], (8*(2**layerNum), 72*(2**layerNum))))
            dataArray2 = np.transpose(np.reshape(weight_layer[:,1], (8*(2**layerNum), 72*(2**layerNum))))
            dataArray3 = np.transpose(np.reshape(weight_layer[:,2], (8*(2**layerNum), 72*(2**layerNum))))
            if i == 1:
                print(dataArray1.shape)

            _, basis1 = pcaTool.pca_lowcost(dataArray1, 8*(2**layerNum))
            _, basis2 = pcaTool.pca_lowcost(dataArray2, 8*(2**layerNum))
            _, basis3 = pcaTool.pca_lowcost(dataArray3, 8*(2**layerNum))
            basis1 = basis1[:,:8*(2**layerNum)]
            basis2 = basis2[:,:8*(2**layerNum)]
            basis3 = basis3[:,:8*(2**layerNum)]
            _, k = smTool.calc_karcher_subspace(basis1, basis3, 8*(2**layerNum))

            if i == 0: magContainer.append(smTool.calc_magnitude(basis2, k, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis2, k))

        return np.array(magContainer)
    
    def _pca_2ndMag_layerwise_fc_CIFAR(self, pcaTool, smTool, csvFolder, totalSampleNum=100):

        magContainer = []
        print("layer fc")

        for i in tqdm(range(totalSampleNum-1 - 1)):
            _, weight_layerw = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer22/part*.csv"), [i + 1, (i + 1) + 1, (i + 2)])

            dataArray1w = np.transpose(np.reshape(weight_layerw[:,0], (10, 128)))
            if i == 1:
                print(dataArray1w.shape)

            dataArray2w = np.transpose(np.reshape(weight_layerw[:,1], (10, 128)))
            dataArray3w = np.transpose(np.reshape(weight_layerw[:,2], (10, 128)))

            _, basis1 = pcaTool.pca_lowcost(dataArray1w, 10)
            _, basis2 = pcaTool.pca_lowcost(dataArray2w, 10)
            _, basis3 = pcaTool.pca_lowcost(dataArray3w, 10)
            basis1 = basis1[:,:10]
            basis2 = basis2[:,:10]
            basis3 = basis3[:,:10]
            _, k = smTool.calc_karcher_subspace(basis1, basis3, 10)

            if i == 0: magContainer.append(smTool.calc_magnitude(basis2, k, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis2, k))

        return np.array(magContainer)

class Resnet18_Tube1stMag():
    
    def __init__(self):
        pass

    def __call__(self, csvFolder, layerNum, sublayerNum, totalSampleNum):
        
        pcaTool = PCA.WeightPCA()
        smTool = SUBSPACE.SubspaceDiff()
        
        if layerNum == 1:
            result = self._pca_1stMag_tube_ip_CIFAR(pcaTool, smTool, csvFolder, totalSampleNum=totalSampleNum)
        elif layerNum == 6:
            print("calculate magnitude between subspace based on tube vectors are not supported in fc layer!")
            result = None
        else:
            if sublayerNum == 1:
                result = self._pca_1stMag_tube_bbi_CIFAR(pcaTool, smTool, csvFolder, layerNum-1, totalSampleNum=totalSampleNum)
            else:
                result = self._pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, csvFolder, layerNum-1, sublayerNum-1, totalSampleNum=totalSampleNum)

        return result
    
    def _pca_1stMag_tube_ip_CIFAR(self, pcaTool, smTool, csvFolder, totalSampleNum=100):

        magContainer = []
        print()
        print("layer input")
        print("mode : tube")

        for i in tqdm(range(totalSampleNum-1)):
            _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer1/part*.csv"), [i+1, i+2])

            dataArray1 = np.reshape(weight_layer[:,0], (24, 49))
            dataArray2 = np.reshape(weight_layer[:,1], (24, 49))
            if i == 1:
                print(dataArray1.shape)

            _, basis1 = pcaTool.pca_lowcost(dataArray1, 49)
            _, basis2 = pcaTool.pca_lowcost(dataArray2, 49)
            basis1 = basis1[:,:49]
            basis2 = basis2[:,:49]
            if i == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

        return np.array(magContainer)

    def _pca_1stMag_tube_bbi_CIFAR(self, pcaTool, smTool, csvFolder, layerNum, totalSampleNum=100):

        magContainer = []
        print()
        print("layer {}(bb(input))".format(layerNum))
        print("mode : tube")

        for i in tqdm(range(totalSampleNum-1)):
            _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 3)), [i+1, i+2])

            dataArray1 = np.reshape(weight_layer[:,0], (32*(4**layerNum), 9))
            dataArray2 = np.reshape(weight_layer[:,1], (32*(4**layerNum), 9))
            if i == 1:
                print(dataArray1.shape)

            _, basis1 = pcaTool.pca_lowcost(dataArray1, 9)
            _, basis2 = pcaTool.pca_lowcost(dataArray2, 9)
            basis1 = basis1[:,:9]
            basis2 = basis2[:,:9]
            if i == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

        return np.array(magContainer)

    def _pca_1stMag_tube_bb_CIFAR(self, pcaTool, smTool, csvFolder, layerNum, sublayerNum, totalSampleNum=100):

        magContainer = []
        print()
        print("layer {}(bb({}th))".format(layerNum, sublayerNum))
        print("mode : tube")

        for i in tqdm(range(totalSampleNum-1)):
            if sublayerNum == 1:
                _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), [i+1, i+2])
            elif sublayerNum == 2:
                _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)), [i+1, i+2])
            elif sublayerNum == 3:
                _, weight_layer = ParamIO.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), [i+1, i+2])
            
            dataArray1 = np.reshape(weight_layer[:,0], (64*(4**layerNum), 9))
            dataArray2 = np.reshape(weight_layer[:,1], (64*(4**layerNum), 9))
            if i == 1:
                print(dataArray1.shape)

            _, basis1 = pcaTool.pca_lowcost(dataArray1, 9)
            _, basis2 = pcaTool.pca_lowcost(dataArray2, 9)
            basis1 = basis1[:,:9]
            basis2 = basis2[:,:9]
            if i == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

        return np.array(magContainer)

# def pca_1stMag_integrated_MNIST(pcaTool, smTool, csvFolder, n_components):

#     magContainer = []

#     _, dataArray_raw1 = ParamIO.makeDataArray(glob(csvFolder + "/layer1/part*.csv"), 1)
#     _, dataArray_raw2 = ParamIO.makeDataArray(glob(csvFolder + "/layer2/part*.csv"), 1)
#     _, dataArray_raw3 = ParamIO.makeDataArray(glob(csvFolder + "/layer3/part*.csv"), 1)
#     _, dataArray_raw4 = ParamIO.makeDataArray(glob(csvFolder + "/layer4/part*.csv"), 1)

#     for colNum in range(dataArray_raw1.shape[1] - 1):
#         dataArray11 = np.transpose(np.reshape(dataArray_raw1[:,colNum], (2, 9)))
#         dataArray12 = np.transpose(np.reshape(dataArray_raw2[:,colNum], (4, 9)))
#         dataArray13 = np.transpose(np.reshape(dataArray_raw3[:,colNum], (8, 9)))
#         dataArray14 = np.transpose(np.reshape(dataArray_raw4[:,colNum], (40, 9)))
#         dataArray1 = np.concatenate((dataArray11, dataArray12, dataArray13, dataArray14), axis=1)
#         dataArray21 = np.transpose(np.reshape(dataArray_raw1[:,colNum + 1], (2, 9)))
#         dataArray22 = np.transpose(np.reshape(dataArray_raw2[:,colNum + 1], (4, 9)))
#         dataArray23 = np.transpose(np.reshape(dataArray_raw3[:,colNum + 1], (8, 9)))
#         dataArray24 = np.transpose(np.reshape(dataArray_raw4[:,colNum + 1], (40, 9)))
#         dataArray2 = np.concatenate((dataArray21, dataArray22, dataArray23, dataArray24), axis=1)
        
#         _, tmp1 = pcaTool.pca_basic(dataArray1)
#         basis1 = tmp1[:,0:n_components]
#         _, tmp2 = pcaTool.pca_basic(dataArray2)
#         basis2 = tmp2[:,0:n_components]

#         magContainer.append(smTool.calc_magnitude(basis1, basis2))

#     return np.array(magContainer)

#     plt.title("first-magnitude of weight subspace(dim={}, layer=all)".format(n_components))
#     plt.xlabel("each step")
#     plt.ylabel("2nd magnitude")
#     plt.plot(range(len(magContainer)), magContainer)
#     plt.savefig("result/layer_secondmag_{:02d}dim_MNIST.png".format(n_components))
#     plt.show()

# def pca_2ndMag_integrated_MNIST(pcaTool, smTool, csvFolder, n_components):

#     magContainer = []

#     _, dataArray_raw1 = ParamIO.makeDataArray(glob(csvFolder + "/layer1/part*.csv"), 1)
#     _, dataArray_raw2 = ParamIO.makeDataArray(glob(csvFolder + "/layer2/part*.csv"), 1)
#     _, dataArray_raw3 = ParamIO.makeDataArray(glob(csvFolder + "/layer3/part*.csv"), 1)
#     _, dataArray_raw4 = ParamIO.makeDataArray(glob(csvFolder + "/layer4/part*.csv"), 1)

#     for colNum in range(dataArray_raw1.shape[1] - 2):
#         dataArray11 = np.transpose(np.reshape(dataArray_raw1[:,colNum], (2, 9)))
#         dataArray12 = np.transpose(np.reshape(dataArray_raw2[:,colNum], (4, 9)))
#         dataArray13 = np.transpose(np.reshape(dataArray_raw3[:,colNum], (8, 9)))
#         dataArray14 = np.transpose(np.reshape(dataArray_raw4[:,colNum], (40, 9)))
#         dataArray1 = np.concatenate((dataArray11, dataArray12, dataArray13, dataArray14), axis=1)
#         dataArray21 = np.transpose(np.reshape(dataArray_raw1[:,colNum + 1], (2, 9)))
#         dataArray22 = np.transpose(np.reshape(dataArray_raw2[:,colNum + 1], (4, 9)))
#         dataArray23 = np.transpose(np.reshape(dataArray_raw3[:,colNum + 1], (8, 9)))
#         dataArray24 = np.transpose(np.reshape(dataArray_raw4[:,colNum + 1], (40, 9)))
#         dataArray2 = np.concatenate((dataArray21, dataArray22, dataArray23, dataArray24), axis=1)
#         dataArray31 = np.transpose(np.reshape(dataArray_raw1[:,colNum + 2], (2, 9)))
#         dataArray32 = np.transpose(np.reshape(dataArray_raw2[:,colNum + 2], (4, 9)))
#         dataArray33 = np.transpose(np.reshape(dataArray_raw3[:,colNum + 2], (8, 9)))
#         dataArray34 = np.transpose(np.reshape(dataArray_raw4[:,colNum + 2], (40, 9)))
#         dataArray3 = np.concatenate((dataArray31, dataArray32, dataArray33, dataArray34), axis=1)

#         _, tmp1 = pcaTool.pca_basic(dataArray1)
#         basis1 = tmp1[:,0:n_components]
#         _, tmp2 = pcaTool.pca_basic(dataArray2)
#         basis2 = tmp2[:,0:n_components]
#         _, tmp3 = pcaTool.pca_basic(dataArray3)
#         basis3 = tmp3[:,0:n_components]

#         _, k = smTool.calc_karcher_subspace(basis1, basis3, n_components)
#         magContainer.append(smTool.calc_magnitude(k, basis2))

#     return np.array(magContainer)

#     plt.title("second-magnitude of weight subspace(dim={}, layer=all)".format(n_components))
#     plt.xlabel("each step")
#     plt.ylabel("2nd magnitude")
#     plt.plot(range(len(magContainer)), magContainer)
#     plt.savefig("result/layer_secondmag_{:02d}dim_MNIST.png".format(n_components))
#     plt.show()

# def pca_1stMag_layerwise_conv_MNIST(pcaTool, smTool, csvFolder, layerNum):

#     magContainer = []
#     rowContainer = [9, 18, 18]
#     colContainer = [2, 2, 4]

#     _, dataArray_raw = ParamIO.makeDataArray(glob(csvFolder + "/layer{}/part*.csv".format(layerNum)), 1)

#     for colNum in range(int(dataArray_raw.shape[1]) - 1):
#         dataArray1 = np.transpose(np.reshape(dataArray_raw[:,1], (colContainer[layerNum - 1], rowContainer[layerNum - 1])))
#         dataArray2 = np.transpose(np.reshape(dataArray_raw[:,colNum + 1], (colContainer[layerNum - 1], rowContainer[layerNum - 1])))

#         _, basis1 = pcaTool.pca_basic(dataArray1)
#         _, basis2 = pcaTool.pca_basic(dataArray2)
#         basis1 = basis1[:,:colContainer[layerNum - 1]]
#         basis2 = basis2[:,:colContainer[layerNum - 1]]
#         if colNum == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
#         else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

#         if colNum == int(dataArray_raw.shape[1]) - 2 : print(smTool.calc_magnitude(basis1, basis2))

#     return np.array(magContainer)

#     plt.title("first-magnitude of weight subspace(dim=7, layer=all)")
#     plt.xlabel("each step")
#     plt.ylabel("1st magnitude")
#     plt.plot(range(len(magContainer)), magContainer)
#     plt.savefig("result/layer_firstmag_7dim_MNIST_{:04d}.png".format(seed))
#     plt.show()

# def pca_1stMag_tube_conv_MNIST(pcaTool, smTool, csvFolder, layerNum):

#     magContainer = []
#     colContainer = [2, 4, 8]

#     _, dataArray_raw = ParamIO.makeDataArray(glob(csvFolder + "/layer{}/part*.csv".format(layerNum)), 1)

#     for colNum in range(int(dataArray_raw.shape[1]) - 1):
#         dataArray1 = np.reshape(dataArray_raw[:,colNum], (colContainer[layerNum - 1], 9))
#         dataArray2 = np.reshape(dataArray_raw[:,colNum + 1], (colContainer[layerNum - 1], 9))

#         _, basis1 = pcaTool.pca_basic(dataArray1)
#         _, basis2 = pcaTool.pca_basic(dataArray2)
#         basis1 = basis1[:,:colContainer[layerNum - 1]]
#         basis2 = basis2[:,:colContainer[layerNum - 1]]
#         if colNum == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
#         else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

#         if colNum == int(dataArray_raw.shape[1]) - 2 : print(smTool.calc_magnitude(basis1, basis2))

#     return np.array(magContainer)

#     plt.title("first-magnitude of weight subspace(dim=7, layer=all)")
#     plt.xlabel("each step")
#     plt.ylabel("1st magnitude")
#     plt.plot(range(len(magContainer)), magContainer)
#     plt.savefig("result/layer_firstmag_7dim_MNIST_{:04d}.png".format(seed))
#     plt.show()

# def pca_1stMag_layerwise_fc_MNIST(pcaTool, smTool, csvFolder):

#     magContainer = []

#     _, dataArray_raw4 = ParamIO.makeDataArray(glob(csvFolder + "/layer4/part*.csv"), 1)

#     for colNum in range(int(dataArray_raw4.shape[1]) - 1):
#         dataArray1 = np.transpose(np.reshape(dataArray_raw4[:,1], (10, 36)))
#         dataArray2 = np.transpose(np.reshape(dataArray_raw4[:,colNum + 1], (10, 36)))

#         _, basis1 = pcaTool.pca_basic(dataArray1)
#         _, basis2 = pcaTool.pca_basic(dataArray2)
#         basis1 = basis1[:,:10]
#         basis2 = basis2[:,:10]
#         if colNum == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
#         else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

#         if colNum == int(dataArray_raw4.shape[1]) - 2 : print(smTool.calc_magnitude(basis1, basis2))

#     return np.array(magContainer)

#     plt.title("first-magnitude of weight subspace(dim=10, layer=fc)")
#     plt.xlabel("each step")
#     plt.ylabel("1st magnitude")
#     plt.plot(range(len(magContainer)), magContainer)
#     plt.savefig("result/layer_firstmag_10dim_MNIST_{:04d}.png".format(seed))
#     plt.show()