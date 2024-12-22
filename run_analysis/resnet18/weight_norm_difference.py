import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from myResearch.module import ParamIO
from glob import glob
from tqdm import tqdm

class Resnet18_DifferenceNorm():

    def __init__(self):
        pass

    # layer number 1 : input layer
    # layer number 2~5 : each basic block
    # layer number 6 : fc layer
    def __call__(self, seed, csvFolder, layerNum, subLayerNum, interval, totalSampleNum):

        if layerNum == 1:
            result = self._print_ip_difference_norm(seed, csvFolder, interval, totalsampleNum=totalSampleNum)
        elif layerNum == 6:
            result = self._print_fc_difference_norm(seed, csvFolder, interval, totalSampleNum=totalSampleNum)
        else:
            if subLayerNum == 1:
                result = self._print_bbi_difference_norm(seed, csvFolder, layerNum-1, interval, totalSampleNum=totalSampleNum)
            else:
                result = self._print_bb_difference_norm(csvFolder, layerNum-1, subLayerNum-1, interval, totalSampleNum=totalSampleNum)

        return result

    def _print_ip_difference_norm(self, seed, csvFolder, interval, totalSampleNum=1000):

        magContainer = []
        print("layer input")

        for i in tqdm(range(int(totalSampleNum / interval))):
            _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer1/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
            before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
            after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
            magContainer.append(np.linalg.norm(before - after))

        return np.array(magContainer)
    
    def _print_fc_difference_norm(self, seed, csvFolder, interval, totalSampleNum=1000):

        magContainer = []
        print("layer fc")

        for i in tqdm(range(int(totalSampleNum / interval))):
            _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer22/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
            before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
            after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
            magContainer.append(np.linalg.norm(before - after))

        return np.array(magContainer)
    
    def _print_bbi_difference_norm(self, seed, csvFolder, layerNum, interval, totalSampleNum=1000):

        magContainer = []
        print("layer {}(bb input)".format(layerNum))

        for i in tqdm(range(int(totalSampleNum / interval))):
            _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 1)), loadMin=interval*i + 1, loadMax=interval*(i+1))
            before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
            after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
            magContainer.append(np.linalg.norm(before - after))

        return np.array(magContainer)
    
    def _print_bb_difference_norm(seed, csvFolder, layerNum, sublayerNum, interval, totalSampleNum=1000):

        magContainer = []
        print("layer {}(bb{}-th)".format(layerNum, sublayerNum))

        for i in tqdm(range(int(totalSampleNum / interval))):

            if sublayerNum == 1:
                _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), loadMin=interval*i + 1, loadMax=interval*(i+1))
                before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
                after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
                magContainer.append(np.linalg.norm(before - after))
            elif sublayerNum == 2:
                _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)), loadMin=interval*i + 1, loadMax=interval*(i+1))
                before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
                after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
                magContainer.append(np.linalg.norm(before - after))
            elif sublayerNum == 3:
                _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), loadMin=interval*i + 1, loadMax=interval*(i+1))
                before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
                after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
                magContainer.append(np.linalg.norm(before - after))

        return np.array(magContainer)
    
class Resnet18_NormDifference():
    
    def __init__(self):
        pass

    # layer number 1 : input layer
    # layer number 2~5 : each basic block
    # layer number 6 : fc layer
    def __call__(self, seed, csvFolder, layerNum, subLayerNum, interval, totalSampleNum):

        if layerNum == 1:
            result = self._print_ip_norm_difference(seed, csvFolder, interval, totalsampleNum=totalSampleNum)
        elif layerNum == 6:
            result = self._print_fc_norm_difference(seed, csvFolder, interval, totalSampleNum=totalSampleNum)
        else:
            if subLayerNum == 1:
                result = self._print_bbi_norm_difference(seed, csvFolder, layerNum-1, interval, totalSampleNum=totalSampleNum)
            else:
                result = self._print_bb_norm_difference(csvFolder, layerNum-1, subLayerNum-1, interval, totalSampleNum=totalSampleNum)

        return result

    def _print_ip_norm_difference(self, seed, csvFolder, interval, totalSampleNum=1000):

        magContainer = []
        print("layer input")

        for i in tqdm(range(int(totalSampleNum / interval))):
            _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer1/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
            magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))

        return np.array(magContainer)



    def _print_bbi_norm_difference(self, seed, csvFolder, layerNum, interval, totalSampleNum=1000):

        magContainer = []
        print("layer {}(bb input)".format(layerNum))

        for i in tqdm(range(int(totalSampleNum / interval))):
            _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), loadMin=interval*i + 1, loadMax=interval*(i+1))
            magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))

        return np.array(magContainer)

    def _print_bb_norm_difference(self, seed, csvFolder, layerNum, sublayerNum, interval, totalSampleNum=1000):

        magContainer = []
        print("layer {}(bb{}-th)".format(layerNum, sublayerNum))

        for i in tqdm(range(int(totalSampleNum / interval))):

            if sublayerNum == 1:
                _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), loadMin=interval*i + 1, loadMax=interval*(i+1))
                magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))
            elif sublayerNum == 2:
                _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)), loadMin=interval*i + 1, loadMax=interval*(i+1))
                magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))
            elif sublayerNum == 3:
                _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), loadMin=interval*i + 1, loadMax=interval*(i+1))
                magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))

        return np.array(magContainer)

    def _print_fc_norm_difference(self, seed, csvFolder, interval, totalSampleNum=1000):

        magContainer = []
        print("layer fc")

        for i in tqdm(range(int(totalSampleNum / interval))):
            _, weight_layer = ParamIO.makeDataArrayInDimSpline(glob(csvFolder + "/layer22/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
            magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))

        return np.array(magContainer)