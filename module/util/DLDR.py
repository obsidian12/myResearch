import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from glob import glob
from tqdm import tqdm
import ParamIO

class WeightDLDR():

    def __init__(self, partitionFolderList, isCenterized=False):
        self.folderList = []
        self.csvList = []
        for partitionFolder in partitionFolderList:
            self.folderList.append(partitionFolder)
            self.csvList.extend(glob(partitionFolder + '/*.csv'))
        self.isCenterized = isCenterized
        self.centerizeCsvList = []
        if isCenterized:
            self.centerizeCsvList = glob(self.folderList[0] +"_centerized" + '/*.csv')

    # self.folderList, self.csvList 내의 모든 열벡터(한 열벡터가 한 행의 형태로 저장됨)를
    # (한 세트에 대해서) 중심화시키는 메소드
    def centerize(self):
        if self.isCenterized:
            print("Centerized is already done!")
            return
        
        newFolder = self.folderList[0] + "_centerized"
        if not os.path.isdir(newFolder):
            os.mkdir(newFolder)
        
        for idx in tqdm(range(len(self.csvList))):
            _, x = ParamIO.makeDataArrayInSampleNum([self.csvList[idx]]) # 열벡터가 파라미터
            x = np.transpose(x) # 행벡터가 파라미터
            row, column = x.shape
            xNorm = np.sum(x, axis=0)
            xMean = xNorm / row
            xMean = np.expand_dims(xMean, axis=0)
            meanMatrix = np.ones((row, 1)) @ xMean
            x = x - meanMatrix

            newCsvFile = newFolder + "/part{:04d}.csv".format(idx)
            self.centerizeCsvList.append(newCsvFile)

            with open(newCsvFile, "w") as f:
                for i in range(row):
                    for j in range(column):
                        if j != 0: f.write(",")
                        f.write(str(x[i, j]))
                    f.write("\n")
                f.close()

        print("Centerizing complete!")
        self.isCenterized = True

    # 웨이트들의 자기상관행렬의 dual, 즉 dual-auto-correlation matrix(refers to (A^T) @ A)를 계산하는 메소드
    # DLDR은  A @ (A^T) 의 계산이 복잡하기에 (A^T) @ A 를 고유치 분해해서 간접적으로 자기상관행렬의 고유벡터를 얻음
    def getDualACM(self):
        if not self.isCenterized:
            print("need to be Centerized!")
            return None

        dataList = []
        dataArray = None

        for i in tqdm(range(len(self.centerizeCsvList))):
            f = open(self.centerizeCsvList[i], "r")
            while True:
                line = f.readline()
                if not line: break
                tmpList = line.split(",")
                
                returnList = []
                for element in tmpList:
                    if element != "": returnList.append(float(element))
                dataList.append(returnList)
            f.close()

            if i == 0: dataArray = np.array(dataList) @ np.transpose(np.array(dataList))
            else:
                dataArray = dataArray + np.array(dataList) @ np.transpose(np.array(dataList))

            dataList = []

        print("Calculating ACM complete!")
        return dataArray
    
    # getDualACM 메소드를 이용해 실질적으로 DLDR을 행하여 그 기저를 계산하는 메소드
    def getDLDRBasis(self, d):
        if not self.isCenterized:
            print("need to be Centerized!")
            return None
        
        dataArray = self.getDualACM()
        s, V = np.linalg.eig(dataArray)
        s = abs(s)
        V = abs(V)

        W = self._getCenterizedW()

        otb = []
        sum = np.sum(s)
        tmp = 0
        for i in tqdm(range(d)):
            sigma = s[i]
            tmp = tmp + sigma
            v = V[:,i]
            u = (W @ v) / np.sqrt(sigma)
            otb.append(u)
        
        print("Calculating DLDR Basis complete!")
        print("total contribution : {}".format(tmp / sum))
        return otb