import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from tqdm import tqdm
import ParamIO
from scipy.spatial.distance import cdist

class WeightPCA():
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
        U = []
        for i in range(n_components):
            v = V[:,i][:,None]
            U.append(np.squeeze(dataArray@v) / np.sqrt(lambdas[i]))
        return [lambdas[:n_components], np.transpose(np.array(U))]
    
    def pca_fulllowcost(self, csvFileList, n_components): #  just doing PCA(doing eigen composition about auto-correlation matrix)

        dataArray_first = ParamIO.makeDataArrayInSampleNum(csvFileList[0])
        r = dataArray_first@np.transpose(dataArray_first)
        dataNum = r.shape[0]

        for csvFile in tqdm(csvFileList[1:]):
            dataArray = ParamIO.makeDataArrayInSampleNum(csvFile)
            r += dataArray@np.transpose(dataArray)
        lambdas, V = np.linalg.eigh(r)
        lambdas = lambdas[::-1]
        V = V[:, ::-1]
        U = []
        for i in range(n_components):
            v = V[:,i]
            listVec = []
            for csvFile in tqdm(csvFileList):
                listVec.append(np.squeeze(np.transpose(ParamIO.makeDataArrayInSampleNum(csvFile))@v[:,None] / np.sqrt(lambdas[i])))
            U.append(listVec)
        return [lambdas[:n_components], U]

    def pca_Proj_lowcost(self, csvFileList, n_components=3): # after doing PCA, project each data to principle subspace
        _, U = WeightPCA.pca_lowcost(self, csvFileList, n_components)
        dataArray = ParamIO.makeDataArrayInSampleNum(csvFileList[0])
        dataNum = dataArray.shape[0]
        coordSum = np.zeros((n_components, dataNum))
        for i, csvFile in tqdm(enumerate(csvFileList)):
            dataArray = ParamIO.makeDataArrayInSampleNum(csvFile)
            coord = np.zeros((n_components, dataNum))
            for data in range(dataNum):
                for pcNum in range(n_components):
                    coord[pcNum][data] = np.dot(dataArray[data,:], U[pcNum][i])
            coordSum += coord
        
        return coordSum
    
    def pca_Proj(self, dataArray, n_components=3): # after doing PCA, project each data to principle subspace
        tmp = WeightPCA.pca_lowcost(self, dataArray, 100)
        lambdas = tmp[1]
        
        result = []
        for dataNum in range(dataArray.shape[1]):
            coord = []
            for pcNum in range(n_components):
                coord.append(np.dot(dataArray[:,dataNum], lambdas[:,pcNum]))
            result.append(coord)
        
        return np.transpose(np.array(result))
    
        
class WeightKPCA():
    def __init__(self):
        pass

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

        X_first = ParamIO.makeDataArrayInSampleNum(csvFileList[0])
        mat_sq_dists = cdist(X_first, X_first, 'sqeuclidean')
        for csvFile in tqdm(csvFileList[1:]):
            X = ParamIO.makeDataArrayInSampleNum(csvFile)
            mat_sq_dists += cdist(X, X, 'sqeuclidean')
        K = np.exp(-gamma * mat_sq_dists)

        if centralizeFlag:
            N = K.shape[0]
            one_n = np.ones((N, N)) / N
            K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        return K