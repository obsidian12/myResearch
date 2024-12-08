# ********************************************************************************************
# 파일명 : pca_projection.py
# 목적　 : 실제로 PCA를 행하여 부분공간을 계산하는 용도의 run 파일
# 구조 　: 클래스 구조(분할 파일 DLDR 계산, 일반 파일 DLDR 계산, PCA 계산, 푸리에 변환)
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

def pca_projection(pcaTool, csvFolderList, seed, n_components, gamma, kernelFlag=True):

    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for csvFolder in csvFolderList:
        csvFileList = glob(csvFolder + "/layer*/part*.csv")
        _, dataArray = ma.makeDataArray(csvFileList, mode=1)

        if kernelFlag:
            projectedDataArray = pcaTool.rbf_kpca_proj(dataArray, n_components, gamma)
            projectedDataArray = np.transpose(projectedDataArray)
        else:
            projectedDataArray = pcaTool.pca_Proj(dataArray, n_components=n_components)
            projectedDataArray = np.transpose(projectedDataArray)

        if n_components == 2:
            x = projectedDataArray[:,0]
            y = projectedDataArray[:,1]
            color = np.linspace(0, len(x), len(x))
            plt.scatter(x, y, c=color, edgecolor='none', s=3)
            plt.colorbar()
            plt.scatter(x[::int(len(x)/20)], y[::int(len(y)/20)], color=['gainsboro', 'mistyrose', 'pink', 'lightcoral', 'indianred', 'red', 'crimson', 'brown', 'darkred', 'black']*2, s=15)
        elif n_components == 3:
            x = projectedDataArray[:,0]
            y = projectedDataArray[:,1]
            z = projectedDataArray[:,2]
            color = np.linspace(0, len(x), len(x))
            ax.scatter(x, y, z, c=color, edgecolor='none', s=3)
            ax.scatter(x[::int(len(x)/20)], y[::int(len(y)/20)], z[::int(len(z)/20)], color=['gainsboro', 'mistyrose', 'pink', 'lightcoral', 'indianred', 'red', 'crimson', 'brown', 'darkred', 'black']*2, s=30, alpha=1)

    titleText = "weight transition visualization by PCA ( "
    titleText += "dim = {}".format(n_components)
    if kernelFlag : titleText += ", gamma = {}".format(gamma)
    titleText += " )"
    plt.title(titleText)
    if kernelFlag: plt.savefig("./result/kpca_weight_dim{:02d}_{:04d}.png".format(n_components, seed))
    else : plt.savefig("./result/linpca_weight_dim{:02d}_{:04d}.png".format(n_components, seed))
    plt.show()

def pca_projection_layerwise(pcaTool, csvFolder, seed, n_components, layerNum, sublayerNum, gamma, kernelFlag=True):

    if sublayerNum > 2: sln = sublayerNum + 1
    else : sln = sublayerNum

    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if layerNum == 0: csvFileList = glob(csvFolder + "/layer1/part*.csv")
    elif layerNum == 5: csvFileList = glob(csvFolder + "/layer22/part*.csv")
    else: csvFileList = glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + sln - 4))
    
    _, dataArray = ma.makeDataArray(csvFileList, mode=1)
    print(dataArray.shape)

    if kernelFlag:
        projectedDataArray = pcaTool.rbf_kpca_proj(dataArray, n_components, gamma)
        projectedDataArray = np.transpose(projectedDataArray)
    else:
        projectedDataArray = pcaTool.pca_Proj(dataArray, n_components=n_components)
        projectedDataArray = np.transpose(projectedDataArray)

    if n_components == 2:
        x = projectedDataArray[:,0]
        y = projectedDataArray[:,1]
        color = np.linspace(0, len(x), len(x))
        plt.scatter(x, y, c=color, edgecolor='none', s=3)
        plt.colorbar()
        plt.scatter(x[::int(len(x)/20)], y[::int(len(y)/20)], color=['gainsboro', 'mistyrose', 'pink', 'lightcoral', 'indianred', 'red', 'crimson', 'brown', 'darkred', 'black']*2, s=15)
    elif n_components == 3:
        x = projectedDataArray[:,0]
        y = projectedDataArray[:,1]
        z = projectedDataArray[:,2]
        color = np.linspace(0, len(x), len(x))
        ax.scatter(x, y, z, c=color, edgecolor='none', s=3)
        ax.scatter(x[::int(len(x)/20)], y[::int(len(y)/20)], z[::int(len(z)/20)], color=['gainsboro', 'mistyrose', 'pink', 'lightcoral', 'indianred', 'red', 'crimson', 'brown', 'darkred', 'black']*2, s=30, alpha=1)

    titleText = "weight transition visualization by PCA ( "
    titleText += "dim = {}".format(n_components)
    if kernelFlag : titleText += ", gamma = {}".format(gamma)
    titleText += " )"
    plt.title(titleText)
    if kernelFlag: plt.savefig("./result/kpca_layer{:02d}{:02d}_dim{:02d}_{:04d}.png".format(layerNum, sublayerNum, n_components, seed))
    else : plt.savefig("./result/linpca_layer{:02d}{:02d}_dim{:02d}_{:04d}.png".format(layerNum, sublayerNum, n_components, seed))
    plt.show()

def pca_projection_lowcost(pcaTool, csvFolderList, seed, n_components, gamma, kernelFlag=True):

    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for csvFolder in csvFolderList:
        csvFileList = glob(csvFolder + "/layer*/part*.csv")

        if kernelFlag:
            projectedDataArray = pcaTool.rbf_kpca_proj_lowcost(csvFileList, n_components, gamma)
            projectedDataArray = np.transpose(projectedDataArray)
        else:
            projectedDataArray = pcaTool.pca_Proj_lowcost(csvFileList, n_components=n_components)
            projectedDataArray = np.transpose(projectedDataArray)

        if n_components == 2:
            x = projectedDataArray[:,0]
            y = projectedDataArray[:,1]
            color = np.linspace(0, len(x), len(x))
            plt.scatter(x, y, c=color, edgecolor='none', s=3)
            plt.colorbar()
            plt.scatter(x[::int(len(x)/20)], y[::int(len(y)/20)], color=['gainsboro', 'mistyrose', 'pink', 'lightcoral', 'indianred', 'red', 'crimson', 'brown', 'darkred', 'black']*2, s=15)
        elif n_components == 3:
            x = projectedDataArray[:,0]
            y = projectedDataArray[:,1]
            z = projectedDataArray[:,2]
            color = np.linspace(0, len(x), len(x))
            ax.scatter(x, y, z, c=color, edgecolor='none', s=3)
            ax.scatter(x[::int(len(x)/20)], y[::int(len(y)/20)], z[::int(len(z)/20)], color=['gainsboro', 'mistyrose', 'pink', 'lightcoral', 'indianred', 'red', 'crimson', 'brown', 'darkred', 'black']*2, s=30, alpha=1)

    titleText = "weight transition visualization by PCA ( "
    titleText += "dim = {}".format(n_components)
    if kernelFlag : titleText += ", gamma = {}".format(gamma)
    titleText += " )"
    plt.title(titleText)
    if kernelFlag: plt.savefig("./result/kpca_weight_dim{:02d}_{:04d}.png".format(n_components, seed))
    else : plt.savefig("./result/linpca_weight_dim{:02d}_{:04d}.png".format(n_components, seed))
    plt.show()


# ------------------------------------------------------------------

n_components = 3 # 사용하는 주성분의 수, 즉 부분공간의 차원
gamma = 5 # kpca를 사용할 때 커널함수의 하이퍼파라미터
pcaTool = wp.weightPCA()
kernelFlag = False # kpca를 사용할 것인지 아닌지를 선택

seed = 1827
csvFolder = "C:/Users/dmtsa/research/run_analysis/DB_RESNET18_CIFAR_8/resnet18__{:04d}".format(seed)

pca_projection_layerwise(pcaTool, csvFolder, seed, n_components, 4, 2, gamma, kernelFlag=kernelFlag)