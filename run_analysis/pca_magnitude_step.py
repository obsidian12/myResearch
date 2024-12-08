# ********************************************************************************************
# 파일명 : pca_magnitude_step.py
# 목적　 : 복수의 weight trajectory에 대해 ***step마다의 부분공간***을 계산,
#          특정 모델의 weight의 일반적인 부분공간이 step이 경과함에 따라 어떻게 변화하는지를 분석하는 모듈(별로 의미 없음 1)
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

def pca_1stMag_step(pcaTool, smTool, csvFolderList, n_components, totalStepNum):

    magContainer = []

    for stepNum in tqdm(range(totalStepNum - 1)):
        _, dataArray1 = ma.makeDataArrayInSet(csvFolderList, stepNum)
        _, dataArray2 = ma.makeDataArrayInSet(csvFolderList, stepNum + 1)

        _, tmp1 = pcaTool.pca_basic(dataArray1)
        basis1 = tmp1[:,0:n_components]
        _, tmp2 = pcaTool.pca_basic(dataArray2)
        basis2 = tmp2[:,0:n_components]
        magContainer.append(smTool.calc_magnitude(basis1, basis2))

    plt.title("first-magnitude of weight subspace(sample-index axis, dim={})".format(n_components))
    plt.xlabel("sample index")
    plt.ylabel("1st magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/step_firstmag_{:02d}dim.png".format(n_components))
    plt.show()

def pca_1stkMag_step(pcaTool, smTool, csvFolderList, n_components, totalStepNum, gamma):

    magContainer = []

    for stepNum in tqdm(range(totalStepNum - 1)):
        _, dataArray1 = ma.makeDataArrayInSet(csvFolderList, stepNum)
        _, dataArray2 = ma.makeDataArrayInSet(csvFolderList, stepNum + 1)

        _, alphas1 = pcaTool.rbf_kernel_pca(dataArray1, n_components, gamma)
        _, alphas2 = pcaTool.rbf_kernel_pca(dataArray2, n_components, gamma)
        km = pcaTool.make_K_matrix(dataArray1, dataArray2, gamma, centralizeFlag=False)
        magContainer.append(smTool.calc_rbf_magnitude(alphas1, alphas2, km))

    plt.title("first-kernel magnitude of weight subspace" + "\n" + "(sample-index axis, dim={}, gamma={})".format(n_components, gamma))
    plt.xlabel("sample index")
    plt.ylabel("1st magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/step_firstkmag_{:02d}dim.png".format(n_components))
    plt.show()
        
def pca_2ndMag_step(pcaTool, smTool, csvFolderList, n_components, totalStepNum):

    magContainer = []

    for stepNum in tqdm(range(totalStepNum - 2)):
        _, dataArray1 = ma.makeDataArrayInSet(csvFolderList, stepNum)
        _, dataArray2 = ma.makeDataArrayInSet(csvFolderList, stepNum + 1)
        _, dataArray3 = ma.makeDataArrayInSet(csvFolderList, stepNum + 2)

        _, tmp1 = pcaTool.pca_basic(dataArray1)
        basis1 = tmp1[:,0:n_components]
        _, tmp2 = pcaTool.pca_basic(dataArray2)
        basis2 = tmp2[:,0:n_components]
        _, tmp3 = pcaTool.pca_basic(dataArray3)
        basis3 = tmp3[:,0:n_components]

        _, k = smTool.calc_karcher_subspace(basis1, basis3, n_components)
        magContainer.append(smTool.calc_magnitude(k, basis2))

    plt.title("second-magnitude of weight subspace(sample-index axis, dim={})".format(n_components))
    plt.xlabel("sample index")
    plt.ylabel("2nd magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/step_secondmag_{:02d}dim.png".format(n_components))
    plt.show()

def pca_2ndkMag_step(pcaTool, smTool, csvFolderList, n_components, totalStepNum, gamma):

    magContainer = []

    for stepNum in tqdm(range(totalStepNum - 2)):
        _, dataArray1 = ma.makeDataArrayInSet(csvFolderList, stepNum)
        _, dataArray2 = ma.makeDataArrayInSet(csvFolderList, stepNum + 1)
        _, dataArray3 = ma.makeDataArrayInSet(csvFolderList, stepNum + 2)

        _, tmp = pcaTool.rbf_kernel_pca_sum(dataArray1, dataArray3, n_components, gamma)
        epsilons = tmp[:,:n_components]
        _, alphas = pcaTool.rbf_kernel_pca(dataArray2, n_components, gamma)
        km = pcaTool.make_K_matrix(np.concatenate((dataArray1, dataArray3), axis=1), dataArray2, gamma, centralizeFlag=False)

        magContainer.append(smTool.calc_rbf_magnitude(epsilons, alphas, km))

    plt.title("second-kernel magnitude of weight subspace" + "\n" + "(sample-index axis, dim={}, gamma={})".format(n_components, gamma))
    plt.xlabel("sample index")
    plt.ylabel("2nd magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/step_secondkmag_{:02d}dim.png".format(n_components))
    plt.show()