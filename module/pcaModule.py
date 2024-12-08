# ********************************************************************************************
# 파일명 : pcaModule.py
# 목적　 : DLDR 기저의 계산, PCA, KPCA 계산, 웨이트의 푸리에 변환 등이 정의된 모듈
# 구조 　: 클래스 구조(분할 파일 DLDR 계산, 일반 파일 DLDR 계산, PCA 계산, 푸리에 변환)
# ********************************************************************************************
import os, sys
#import makeArray as ma
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy as cp
from glob import glob
from tqdm import tqdm
#import makeArray as ma
from scipy.spatial.distance import pdist, cdist, squareform

def tmp(csvFile): # 여기서 FileList란 학습 한번에 저장된 웨이트 파일 리스트를 말하는 거임
    
    dataList = []
    f = open(csvFile, "r")

    while True:
        line = f.readline()
        if not line: break
        tmpList = line.split(",")
        returnList = []
        for element in tmpList:
            if element != "": returnList.append(float(element))
        dataList.append(returnList)
    f.close()

    dataArray = np.array(dataList)
    return dataArray

def makeDataArrayInDim(csvFileList, loadMin=0, loadMax=1048576): # 여기서 FileList란 학습 한번에 저장된 웨이트 파일 리스트를 말하는 거임
    unitNumCounter = 0 # in this mode, unitNumCounter equals to number of column of the matrix
    tmpCounter = 0
    dataList = []

    for csvFile in csvFileList:
        f = open(csvFile, "r")
        tmpCounter = 0
        while True:
            line = f.readline()
            if tmpCounter <= loadMin: continue
            if not line or not (tmpCounter < loadMax): break
            if csvFileList.index(csvFile) == 0: 
                unitNumCounter = unitNumCounter + 1
                tmpList = line.split(",")
                returnList = []
                for element in tmpList:
                    if element != "": returnList.append(float(element))
                dataList.append(returnList)
            else:
                tmpList = line.split(",")
                for element in tmpList:
                    if element != "": dataList[tmpCounter].append(float(element))
            tmpCounter = tmpCounter + 1
        f.close()

    dataArray = np.array(dataList)
    return unitNumCounter, np.transpose(dataArray)

class weightDLDR_divided():

    def __init__(self, partitionFolderList, isCenterized=False):
        self.folderList = [] # 
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
           # _, x = makeDataArray([self.csvList[idx]]) # 열벡터가 파라미터
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
        dataArray = np.arange(1, 17).reshape(4,4)

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

class weightDLDR_integrated():
# def makeDualACM(csvFile, splitNum=1):
    
#     lengthGetFlag = True
#     length = 0
#     splitUnit = 0

#     dataList = []

#     f = open(csvFile, "r")
#     while True:
#         line = f.readline()
#         if not line: break
#         tmpList = line.split(",")

#         if lengthGetFlag:
#             length = len(tmpList)
#             splitUnit = length / splitNum
#             lengthGetFlag = False
        
#         returnList = []
#         for i in range(splitUnit):
#             if tmpList[i] != "": returnList.append(float(tmpList[i]))
#         dataList.append(returnList)
#     f.close()

#     dataArray = np.array(dataList) @ np.transpose(np.array(dataList))

#     for i in range(1, splitNum):
#         f = open(csvFile, "r")
#         while True:
#             line = f.readline()
#             if not line: break
#             tmpList = line.split(",")
            
#             returnList = []
#             for element in tmpList[i * splitUnit:(i + 1) * splitUnit]:
#                     if element != "": returnList.append(float(element))
#             dataList.append(returnList)

#         dataArray = dataArray + np.array(dataList) @ np.transpose(np.array(dataList))

#     return dataArray

# def DLDR(param_trajectory, d):
#     t = param_trajectory.shape[1]
#     w_mean = param_trajectory[:,0]

#     for i in range(1, t):
#         w_mean = w_mean + param_trajectory[:,i]
#     w_mean = w_mean / t

#     W = cp.deepcopy(param_trajectory)
#     for i in range(t):
#         W[:,i] = W[:,i] - w_mean

#     s, V = np.linalg.eig(np.transpose(W) @ W)
#     s = abs(s)
#     V = abs(V)

#     otb = []
#     sum = np.sum(s)
#     tmp = 0
#     for i in range(d):
#         sigma = s[i]
#         tmp = tmp + sigma
#         v = V[:,i]
#         u = (W @ v) / np.sqrt(sigma)
#         otb.append(u)
    
#     print("total contribution : {}".format(tmp / sum))
#     return otb
    pass

class weightPCA():
    def __init__(self):
        pass

    def pca_basic(self, dataArray): #  just doing PCA(doing eigen composition about auto-correlation matrix)
        r = dataArray@np.transpose(dataArray)
        alphas, lambdas = np.linalg.eigh(r)
        return [alphas[::-1], lambdas[:, ::-1]]
    
    def pca_lowcost(self, dataArray, n_components): #  just doing PCA(doing eigen composition about auto-correlation matrix)

        r = np.transpose(dataArray)@dataArray
        lambdas, V = np.linalg.eigh(r)
        lambdas = lambdas[::-1]
        V = V[:, ::-1]
        # for i in range(10):
        #     print("cumulative contribution to {}-th component : ".format(i + 1), end="")
        #     print(np.sum(lambdas[:i+1]) / np.sum(lambdas) * 100, end="")
        #     print("%")
        U = []
        for i in range(n_components):
            v = V[:,i][:,None]
            U.append(np.squeeze(dataArray@v) / np.sqrt(lambdas[i]))
        return [lambdas[:n_components], np.transpose(np.array(U))]
    
    def pca_fulllowcost(self, csvFileList, n_components): #  just doing PCA(doing eigen composition about auto-correlation matrix)

        dataArray_first = tmp(csvFileList[0])
        r = dataArray_first@np.transpose(dataArray_first)
        dataNum = r.shape[0]

        for csvFile in tqdm(csvFileList[1:]):
            dataArray = tmp(csvFile)
            r += dataArray@np.transpose(dataArray)
        lambdas, V = np.linalg.eigh(r)
        lambdas = lambdas[::-1]
        V = V[:, ::-1]
        U = []
        for i in range(n_components):
            v = V[:,i]
            listVec = []
            for csvFile in tqdm(csvFileList):
                listVec.append(np.squeeze(np.transpose(tmp(csvFile))@v[:,None] / np.sqrt(lambdas[i])))
            U.append(listVec)
        return [lambdas[:n_components], U]

    def pca_Proj_lowcost(self, csvFileList, n_components=3): # after doing PCA, project each data to principle subspace
        _, U = weightPCA.pca_lowcost(self, csvFileList, n_components)
        dataArray = tmp(csvFileList[0])
        dataNum = dataArray.shape[0]
        coordSum = np.zeros((n_components, dataNum))
        for i, csvFile in tqdm(enumerate(csvFileList)):
            dataArray = tmp(csvFile)
            coord = np.zeros((n_components, dataNum))
            for data in range(dataNum):
                for pcNum in range(n_components):
                    coord[pcNum][data] = np.dot(dataArray[data,:], U[pcNum][i])
            coordSum += coord
        
        return coordSum
    
    def pca_Proj(self, dataArray, n_components=3): # after doing PCA, project each data to principle subspace
        tmp = weightPCA.pca_lowcost(self, dataArray, 100)
        # s = 0
        # for i, a in enumerate(tmp[0]):
        #     s += a
        #     print("{}-th sum : {} || {}".format(i+1, s * 100 / 28956, a))
        lambdas = tmp[1]
        
        result = []
        for dataNum in range(dataArray.shape[1]):
            coord = []
            for pcNum in range(n_components):
                coord.append(np.dot(dataArray[:,dataNum], lambdas[:,pcNum]))
            result.append(coord)
        
        return np.transpose(np.array(result))
    
    def rbf_kpca_proj(self, dataArray, n_components, gamma=5): # after doing PCA, project each data to principle subspace
        """
        RBF 커널 부분공간 정사영 구현

        매개변수
        ------------
        dataArray: {넘파이 ndarray}, shape = [n_features, n_samples]
        부분공간으로 정사영할 (원본공간) datapoint 들 묶음
         
        gamma: float
        RBF 커널 튜닝 매개변수
            
        n_components: int
        사용할 주성분 개수

        Returns
        ------------
        l: list
        고윳값

        alphas: {넘파이 ndarray}, shape = [n_samples, k_features]
        각 열 : 고유벡터의 datapoint 에 대한 결합계수 인 행렬
        

        고차원 공간을 다루는 특성상 각 주성분을 직접 구하지 못하고,
        datapoint 들에 대한 결합계수로써 밖에 구할 수 없음에 주의한다.
        """
        tmp = self.rbf_kernel_pca(dataArray, n_components, gamma)
        alphas = tmp[1]
        
        result = []
        for dataNum in range(dataArray.shape[1]): # iteration : 각 투영할 datapoint
            coord = []
            for pcNum in range(n_components): # iteration : 각 좌표(주성분)
                a = alphas[:,pcNum]
                sum = 0
                for elementNum in range(dataArray.shape[1]):
                    sum += a[elementNum] * self.rbf_kernel(dataArray[:,dataNum], dataArray[:,elementNum], gamma)
                coord.append(sum)
            result.append(coord)
        
        return np.transpose(np.array(result))
    
    def rbf_kpca_proj_lowcost(self, csvFileList, n_components, gamma=5): # after doing PCA, project each data to principle subspace
        """
        RBF 커널 부분공간 정사영 구현

        매개변수
        ------------
        dataArray: {넘파이 ndarray}, shape = [n_features, n_samples]
        부분공간으로 정사영할 (원본공간) datapoint 들 묶음
         
        gamma: float
        RBF 커널 튜닝 매개변수
            
        n_components: int
        사용할 주성분 개수

        Returns
        ------------
        l: list
        고윳값

        alphas: {넘파이 ndarray}, shape = [n_samples, k_features]
        각 열 : 고유벡터의 datapoint 에 대한 결합계수 인 행렬
        

        고차원 공간을 다루는 특성상 각 주성분을 직접 구하지 못하고,
        datapoint 들에 대한 결합계수로써 밖에 구할 수 없음에 주의한다.
        """
        returnList = self.rbf_kernel_pca_lowcost(csvFileList, n_components, gamma)
        alphas = returnList[1]
        K = returnList[2]
        
        result = []
        for dataNum in range(K.shape[1]): # iteration : 각 투영할 datapoint
            coord = []
            for pcNum in range(n_components): # iteration : 각 좌표(주성분)
                a = alphas[:,pcNum]
                sum = np.dot(a, K[:,dataNum])
                coord.append(sum)
            result.append(coord)
        
        return np.transpose(np.array(result))
    
    def rbf_kernel_pca(self, dataArray, n_component, gamma):
        """
        RBF 커널 PCA 구현

        매개변수
        ------------
        X: {넘파이 ndarray}, shape = [n_samples, n_features]
            
        gamma: float
        RBF 커널 튜닝 매개변수

        Returns
        ------------
        l: list
        고윳값

        alphas: {넘파이 ndarray}, shape = [n_samples, k_features]
        각 열 : 고유벡터의 datapoint 에 대한 결합계수 인 행렬
        

        고차원 공간을 다루는 특성상 각 주성분을 직접 구하지 못하고,
        datapoint 들에 대한 결합계수로써 밖에 구할 수 없음에 주의한다.
        """
        # calculate kernel-matrix(K-matrix).
        K = self.make_K_matrix(dataArray, dataArray, gamma)

        # eigen-decompose centralized K-matrix.
        N = K.shape[0]
        l, alphas = np.linalg.eigh(K)
        alphasLen = np.linalg.norm(alphas, axis=0) * np.sqrt(N*l)
        alphas = alphas / alphasLen
        l = l[::-1]
        alphas = alphas[:, ::-1]
        return [l[:n_component], alphas[:,:n_component]]
    
    def rbf_kernel_pca_lowcost(self, csvFileList, n_component, gamma):
        """
        RBF 커널 PCA 구현

        매개변수
        ------------
        X: {넘파이 ndarray}, shape = [n_samples, n_features]
            
        gamma: float
        RBF 커널 튜닝 매개변수

        Returns
        ------------
        l: list
        고윳값

        alphas: {넘파이 ndarray}, shape = [n_samples, k_features]
        각 열 : 고유벡터의 datapoint 에 대한 결합계수 인 행렬
        

        고차원 공간을 다루는 특성상 각 주성분을 직접 구하지 못하고,
        datapoint 들에 대한 결합계수로써 밖에 구할 수 없음에 주의한다.
        """
        # calculate kernel-matrix(K-matrix).
        K = self.make_K_matrix_lowcost(csvFileList, gamma)

        # eigen-decompose centralized K-matrix.
        N = K.shape[0]
        l, alphas = np.linalg.eigh(K)
        alphasLen = np.linalg.norm(alphas, axis=0) * np.sqrt(N*l)
        alphas = alphas / alphasLen
        l = l[::-1]
        alphas = alphas[:, ::-1]
        return [l[:n_component], alphas[:,:n_component], K]
    
    def rbf_kernel_pca_sum(self, dataArray1, dataArray2, n_components, gamma):
        """
        RBF 커널 PCA 구현(합공간)

        매개변수
        ------------
        X: {넘파이 ndarray}, shape = [n_samples, n_features]
            
        gamma: float
        RBF 커널 튜닝 매개변수

        Returns
        ------------
        l: list
        고윳값

        alphas: {넘파이 ndarray}, shape = [n_samples, k_features]
        각 열 : 고유벡터의 datapoint 에 대한 결합계수 인 행렬
        

        고차원 공간을 다루는 특성상 각 주성분을 직접 구하지 못하고,
        datapoint 들에 대한 결합계수로써 밖에 구할 수 없음에 주의한다.
        """
        # calculate kernel-matrix(K-matrix).
        _, tmp1 = self.rbf_kernel_pca(dataArray1, n_components, gamma)
        tmp_alphas1 = tmp1[:,0:n_components]
        _, tmp2 = self.rbf_kernel_pca(dataArray2, n_components, gamma)
        tmp_alphas2 = tmp2[:,0:n_components]
        K = self.make_K_matrix(np.concatenate((dataArray1, dataArray2), axis=1), np.concatenate((dataArray1, dataArray2), axis=1), gamma)

        # 
        O = np.zeros_like(tmp_alphas1)
        alphas1 = np.concatenate((tmp_alphas1, O), axis=0)
        alphas2 = np.concatenate((O, tmp_alphas2), axis=0)
        a = np.expand_dims(alphas1[:,0], axis=1)
        CA = a @ np.transpose(a)
        for i in range(alphas1.shape[1]- 1):
            a = np.expand_dims(alphas1[:,i], axis=1)
            CA = CA + a @ np.transpose(a)
        for i in range(alphas2.shape[1]):
            a = np.expand_dims(alphas2[:,i], axis=1)
            CA = CA + a @ np.transpose(a)
        
        # eigen-decompose centralized K-matrix.
        l, epsilons = np.linalg.eigh(CA @ K)
        l = l[::-1]
        epsilons = epsilons[:, ::-1]
        return [l[:2*n_components], epsilons[:,:2*n_components]]
    
    def rbf_kernel(self, x_1, x_2, gamma):
        dist = np.linalg.norm(x_1 - x_2)
        return np.exp(-gamma * dist * dist)
    
    # calculate kernel-matrix(K-matrix).
    # (1. calculate squre of euclidean distance of all sample pair.)
    # (2. convert distance of all sample pair to square form)
    # (3. calculate value of RBF kernel (exp(-gamma * ||x - x||^2)) about each element of distance matrix)
    def make_K_matrix(self, dataArray1, dataArray2, gamma, centralizeFlag=True):
        X1 = np.transpose(dataArray1)
        X2 = np.transpose(dataArray2)
        mat_sq_dists = cdist(X1, X2, 'sqeuclidean')
        K = np.exp(-gamma * mat_sq_dists)

        if centralizeFlag:
            N = K.shape[0]
            one_n = np.ones((N, N)) / N
            K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        return K
    
    def make_K_matrix_lowcost(self, csvFileList, gamma, centralizeFlag=True):

        X_first = tmp(csvFileList[0])
        mat_sq_dists = cdist(X_first, X_first, 'sqeuclidean')
        for csvFile in tqdm(csvFileList[1:]):
            X = tmp(csvFile)
            mat_sq_dists += cdist(X, X, 'sqeuclidean')
        K = np.exp(-gamma * mat_sq_dists)

        if centralizeFlag:
            N = K.shape[0]
            one_n = np.ones((N, N)) / N
            K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        return K

    
class weightFFT():

    def __init__(self):
        pass

    @staticmethod
    def fft(dataVec, sampleFreq):
        n = len(dataVec)
        k = np.arange(n)
        freqSpace = k * sampleFreq / n
        Y = np.fft.fft(dataVec)/n
        return (freqSpace, Y)

