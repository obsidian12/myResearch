# ********************************************************************************************
# 파일명 : pca_magnitude_integrated.py
# 목적　 : 1개의 weight trajectory에 대해 ***step 별로 weight 전체의 부분공간***을 계산,
#          step 별로 weight 전체의 부분공간이 어떻게 변화하는지 분석하는 모듈
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

def pca_1stMag_integrated_CIFAR(pcaTool, smTool, seed, csvFolder, n_components):

    magContainer = []

    for i in tqdm(range(1, 1000)):
        _, weight_layer01 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer1/part*.csv"), loadMin=i, loadMax=i + 2)
        _, weight_layer02 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer2/part*.csv") + glob(csvFolder + "/layer3/part*.csv") + glob(csvFolder + "/layer4/part*.csv"), loadMin=i, loadMax=i + 2)
        _, weight_layer03 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer5/part*.csv") + glob(csvFolder + "/layer6/part*.csv"), i, i + 2)
        _, weight_layer04 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer7/part*.csv") + glob(csvFolder + "/layer8/part*.csv") + glob(csvFolder + "/layer9/part*.csv"), loadMin=i, loadMax=i + 2)
        _, weight_layer05 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer10/part*.csv") + glob(csvFolder + "/layer11/part*.csv"), i, i + 2)
        _, weight_layer06 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer12/part*.csv") + glob(csvFolder + "/layer13/part*.csv") + glob(csvFolder + "/layer14/part*.csv"), loadMin=i, loadMax=i + 2)
        _, weight_layer07 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer15/part*.csv") + glob(csvFolder + "/layer16/part*.csv"), i, i + 2)
        _, weight_layer08 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer17/part*.csv") + glob(csvFolder + "/layer18/part*.csv") + glob(csvFolder + "/layer19/part*.csv"), loadMin=i, loadMax=i + 2)
        _, weight_layer09 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer20/part*.csv") + glob(csvFolder + "/layer21/part*.csv"), i, i + 2)
        _, weight_layer10 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer22/part*.csv") + glob(csvFolder + "/layer23/part*.csv"), i, i + 2)

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

    # plt.title("first-magnitude of weight subspace(dim={}, total)".format(n_components))
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/total_firstmag_{:02d}dim_{:04d}.png".format(n_components, seed))
    # # plt.show()
    # plt.clf()

    return np.array(magContainer)

def pca_1stMag_integrated_MNIST(pcaTool, smTool, csvFolder, n_components):

    magContainer = []

    _, dataArray_raw1 = ma.makeDataArray(glob(csvFolder + "/layer1/part*.csv"), 1)
    _, dataArray_raw2 = ma.makeDataArray(glob(csvFolder + "/layer2/part*.csv"), 1)
    _, dataArray_raw3 = ma.makeDataArray(glob(csvFolder + "/layer3/part*.csv"), 1)
    _, dataArray_raw4 = ma.makeDataArray(glob(csvFolder + "/layer4/part*.csv"), 1)

    for colNum in range(dataArray_raw1.shape[1] - 1):
        dataArray11 = np.transpose(np.reshape(dataArray_raw1[:,colNum], (2, 9)))
        dataArray12 = np.transpose(np.reshape(dataArray_raw2[:,colNum], (4, 9)))
        dataArray13 = np.transpose(np.reshape(dataArray_raw3[:,colNum], (8, 9)))
        dataArray14 = np.transpose(np.reshape(dataArray_raw4[:,colNum], (40, 9)))
        dataArray1 = np.concatenate((dataArray11, dataArray12, dataArray13, dataArray14), axis=1)
        dataArray21 = np.transpose(np.reshape(dataArray_raw1[:,colNum + 1], (2, 9)))
        dataArray22 = np.transpose(np.reshape(dataArray_raw2[:,colNum + 1], (4, 9)))
        dataArray23 = np.transpose(np.reshape(dataArray_raw3[:,colNum + 1], (8, 9)))
        dataArray24 = np.transpose(np.reshape(dataArray_raw4[:,colNum + 1], (40, 9)))
        dataArray2 = np.concatenate((dataArray21, dataArray22, dataArray23, dataArray24), axis=1)
        
        _, tmp1 = pcaTool.pca_basic(dataArray1)
        basis1 = tmp1[:,0:n_components]
        _, tmp2 = pcaTool.pca_basic(dataArray2)
        basis2 = tmp2[:,0:n_components]

        magContainer.append(smTool.calc_magnitude(basis1, basis2))

    return np.array(magContainer)

    plt.title("first-magnitude of weight subspace(dim={}, layer=all)".format(n_components))
    plt.xlabel("each step")
    plt.ylabel("2nd magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/layer_secondmag_{:02d}dim_MNIST.png".format(n_components))
    plt.show()

def pca_2ndMag_integrated_MNIST(pcaTool, smTool, csvFolder, n_components):

    magContainer = []

    _, dataArray_raw1 = ma.makeDataArray(glob(csvFolder + "/layer1/part*.csv"), 1)
    _, dataArray_raw2 = ma.makeDataArray(glob(csvFolder + "/layer2/part*.csv"), 1)
    _, dataArray_raw3 = ma.makeDataArray(glob(csvFolder + "/layer3/part*.csv"), 1)
    _, dataArray_raw4 = ma.makeDataArray(glob(csvFolder + "/layer4/part*.csv"), 1)

    for colNum in range(dataArray_raw1.shape[1] - 2):
        dataArray11 = np.transpose(np.reshape(dataArray_raw1[:,colNum], (2, 9)))
        dataArray12 = np.transpose(np.reshape(dataArray_raw2[:,colNum], (4, 9)))
        dataArray13 = np.transpose(np.reshape(dataArray_raw3[:,colNum], (8, 9)))
        dataArray14 = np.transpose(np.reshape(dataArray_raw4[:,colNum], (40, 9)))
        dataArray1 = np.concatenate((dataArray11, dataArray12, dataArray13, dataArray14), axis=1)
        dataArray21 = np.transpose(np.reshape(dataArray_raw1[:,colNum + 1], (2, 9)))
        dataArray22 = np.transpose(np.reshape(dataArray_raw2[:,colNum + 1], (4, 9)))
        dataArray23 = np.transpose(np.reshape(dataArray_raw3[:,colNum + 1], (8, 9)))
        dataArray24 = np.transpose(np.reshape(dataArray_raw4[:,colNum + 1], (40, 9)))
        dataArray2 = np.concatenate((dataArray21, dataArray22, dataArray23, dataArray24), axis=1)
        dataArray31 = np.transpose(np.reshape(dataArray_raw1[:,colNum + 2], (2, 9)))
        dataArray32 = np.transpose(np.reshape(dataArray_raw2[:,colNum + 2], (4, 9)))
        dataArray33 = np.transpose(np.reshape(dataArray_raw3[:,colNum + 2], (8, 9)))
        dataArray34 = np.transpose(np.reshape(dataArray_raw4[:,colNum + 2], (40, 9)))
        dataArray3 = np.concatenate((dataArray31, dataArray32, dataArray33, dataArray34), axis=1)

        _, tmp1 = pcaTool.pca_basic(dataArray1)
        basis1 = tmp1[:,0:n_components]
        _, tmp2 = pcaTool.pca_basic(dataArray2)
        basis2 = tmp2[:,0:n_components]
        _, tmp3 = pcaTool.pca_basic(dataArray3)
        basis3 = tmp3[:,0:n_components]

        _, k = smTool.calc_karcher_subspace(basis1, basis3, n_components)
        magContainer.append(smTool.calc_magnitude(k, basis2))

    return np.array(magContainer)

    plt.title("second-magnitude of weight subspace(dim={}, layer=all)".format(n_components))
    plt.xlabel("each step")
    plt.ylabel("2nd magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/layer_secondmag_{:02d}dim_MNIST.png".format(n_components))
    plt.show()