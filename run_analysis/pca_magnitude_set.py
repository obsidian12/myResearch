# ********************************************************************************************
# 파일명 : pca_magnitude_set.py
# 목적　 : 복수의 weight trajectory에 대해 ***각 trajectory마다의 부분공간***을 계산,
#          trajectory 마다의 부분공간이 얼마나 멀리 떨어져 있는지를 분석하는 모듈(별로 의미 없음 2)
# 구조 　: 함수 구조(1st magnitude, 2nd magnitude)
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

def pca_1stMag_set(pcaTool, smTool, csvFolderList, n_components):

    magContainer = []

    for setNum in tqdm(range(len(csvFolderList) - 1)):

        _, dataArray1 = ma.makeDataArray(glob(csvFolderList[setNum] + "/layer*/part*.csv"), 1)
        _, dataArray2 = ma.makeDataArray(glob(csvFolderList[setNum + 1] + "/layer*/part*.csv"), 1)

        _, tmp1 = pcaTool.pca_basic(dataArray1)
        basis1 = tmp1[:,0:n_components]
        _, tmp2 = pcaTool.pca_basic(dataArray2)
        basis2 = tmp2[:,0:n_components]
        magContainer.append(smTool.calc_magnitude(basis1, basis2))

    plt.title("first-magnitude of weight subspace(set axis, dim={})".format(n_components))
    plt.xlabel("number of each set")
    plt.ylabel("1st magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/set_firstmag_{:02d}dim.png".format(n_components))
    plt.show()
        
def pca_2ndMag_set(pcaTool, smTool, csvFolderList, n_components):

    magContainer = []

    for setNum in tqdm(range(len(csvFolderList) - 2)):

        _, dataArray1 = ma.makeDataArray(glob(csvFolderList[setNum] + "/layer*/part*.csv"), 1)
        _, dataArray2 = ma.makeDataArray(glob(csvFolderList[setNum + 1] + "/layer*/part*.csv"), 1)
        _, dataArray3 = ma.makeDataArray(glob(csvFolderList[setNum + 2] + "/layer*/part*.csv"), 1)

        _, tmp1 = pcaTool.pca_basic(dataArray1)
        basis1 = tmp1[:,0:n_components]
        _, tmp2 = pcaTool.pca_basic(dataArray2)
        basis2 = tmp2[:,0:n_components]
        _, tmp3 = pcaTool.pca_basic(dataArray3)
        basis3 = tmp3[:,0:n_components]

        _, k = smTool.calc_karcher_subspace(basis1, basis3, n_components)
        magContainer.append(smTool.calc_magnitude(k, basis2))

    plt.title("second-magnitude of weight subspace(set axis, dim={})".format(n_components))
    plt.xlabel("number of each set")
    plt.ylabel("2nd magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/set_secondmag_{:02d}dim.png".format(n_components))
    plt.show()