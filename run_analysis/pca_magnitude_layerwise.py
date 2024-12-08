# ********************************************************************************************
# 파일명 : pca_magnitude_layerwise.py
# 목적　 : 1개의 weight trajectory에 대해 ***step 별로, layer 별로 weight의 부분공간***을 계산,
#          step 별로 각 층의 weight 부분공간이 어떻게 변화하는지 분석하는 모듈
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
from matplotlib.animation import FuncAnimation


def pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder, interval, totalSampleNum=1000):

    magContainer = []
    print("layer input")

    for i in tqdm(range(int(totalSampleNum / interval))):
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer1/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))

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

        if i == int(totalSampleNum / interval) - 1 : print(smTool.calc_magnitude(basis1, basis2))

    # plt.title("first-magnitude of weight subspace(dim=24, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_24dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def pca_2ndMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder, interval, totalSampleNum=1000):

    magContainer = []
    print("layer input")

    for i in tqdm(range(int(totalSampleNum / interval) - 1)):
        _, weight_layer = ma.makeDataArrayInDimSplines(glob(csvFolder + "/layer1/part*.csv"), [interval*i + 1, interval*(i + 1) + 1, interval*(i + 2)])

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

    # plt.title("first-magnitude of weight subspace(dim=24, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_24dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def pca_1stMag_tube_ip_CIFAR(pcaTool, smTool, seed, csvFolder, interval, totalSampleNum=1000):

    magContainer = []
    print()
    print("layer input")
    print("mode : tube")

    for i in tqdm(range(int(totalSampleNum / interval))):
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer1/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))

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

    # plt.title("first-magnitude of weight subspace(dim=24, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_24dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, layerNum, interval, totalSampleNum=1000):

    magContainer = []
    print("layer {}(bb(input))".format(layerNum))

    for i in tqdm(range(int(totalSampleNum / interval))):
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 3)), loadMin=interval*i + 1, loadMax=interval*(i+1))

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

        if i == int(totalSampleNum / interval) - 1 : print(smTool.calc_magnitude(basis1, basis2))

        # plt.title("first-magnitude of weight subspace(dim={}, layer={})".format(compNumContainer[layerNum - 1], layerNum))
        # plt.xlabel("each step")
        # plt.ylabel("1st magnitude")
        # plt.plot(range(len(magContainer)), magContainer)
        # plt.savefig("result/layer{:02d}_firstmag_{:02d}dim_{:04d}.png".format(layerNum, compNumContainer[layerNum - 1], seed))
        # plt.show()
        # plt.clf()

    return np.array(magContainer)

def pca_2ndMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, layerNum, interval, totalSampleNum=1000):

    magContainer = []
    print("layer {}(bb(input))".format(layerNum))

    for i in tqdm(range(int(totalSampleNum / interval) - 1)):
        _, weight_layer = ma.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 3)), [interval*i + 1, interval*(i + 1) + 1, interval*(i + 2)])

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

        # plt.title("first-magnitude of weight subspace(dim={}, layer={})".format(compNumContainer[layerNum - 1], layerNum))
        # plt.xlabel("each step")
        # plt.ylabel("1st magnitude")
        # plt.plot(range(len(magContainer)), magContainer)
        # plt.savefig("result/layer{:02d}_firstmag_{:02d}dim_{:04d}.png".format(layerNum, compNumContainer[layerNum - 1], seed))
        # plt.show()
        # plt.clf()

    return np.array(magContainer)

def pca_1stMag_tube_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, layerNum, interval, totalSampleNum=1000):

    magContainer = []
    print()
    print("layer {}(bb(input))".format(layerNum))
    print("mode : tube")

    for i in tqdm(range(int(totalSampleNum / interval))):
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 3)), loadMin=interval*i + 1, loadMax=interval*(i+1))

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

        # plt.title("first-magnitude of weight subspace(dim={}, layer={})".format(compNumContainer[layerNum - 1], layerNum))
        # plt.xlabel("each step")
        # plt.ylabel("1st magnitude")
        # plt.plot(range(len(magContainer)), magContainer)
        # plt.savefig("result/layer{:02d}_firstmag_{:02d}dim_{:04d}.png".format(layerNum, compNumContainer[layerNum - 1], seed))
        # plt.show()
        # plt.clf()

    return np.array(magContainer)

def pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, layerNum, sublayerNum, interval, totalSampleNum=1000):

    magContainer = []
    print("layer {}(bb({}th))".format(layerNum, sublayerNum))

    for i in tqdm(range(int(totalSampleNum / interval))):
        if sublayerNum == 1:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), loadMin=interval*i + 1, loadMax=interval*(i+1))
        elif sublayerNum == 2:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)), loadMin=interval*i + 1, loadMax=interval*(i+1))
        elif sublayerNum == 3:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), loadMin=interval*i + 1, loadMax=interval*(i+1))
        
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

        if i == int(totalSampleNum / interval) - 1 : print(smTool.calc_magnitude(basis1, basis2))

        # plt.title("first-magnitude of weight subspace(dim={}, layer={})".format(compNumContainer[layerNum - 1], layerNum))
        # plt.xlabel("each step")
        # plt.ylabel("1st magnitude")
        # plt.plot(range(len(magContainer)), magContainer)
        # plt.savefig("result/layer{:02d}_firstmag_{:02d}dim_{:04d}.png".format(layerNum, compNumContainer[layerNum - 1], seed))
        # plt.show()
        # plt.clf()

    return np.array(magContainer)

def pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, layerNum, sublayerNum, interval, totalSampleNum=1000):

    magContainer = []
    print("layer {}(bb({}th))".format(layerNum, sublayerNum))

    for i in tqdm(range(int(totalSampleNum / interval) - 1)):
        if sublayerNum == 1:
            _, weight_layer = ma.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), [interval*i + 1, interval*(i + 1) + 1, interval*(i + 2)])
        elif sublayerNum == 2:
            _, weight_layer = ma.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)), [interval*i + 1, interval*(i + 1) + 1, interval*(i + 2)])
        elif sublayerNum == 3:
            _, weight_layer = ma.makeDataArrayInDimSplines(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), [interval*i + 1, interval*(i + 1) + 1, interval*(i + 2)])
        
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

        # plt.title("first-magnitude of weight subspace(dim={}, layer={})".format(compNumContainer[layerNum - 1], layerNum))
        # plt.xlabel("each step")
        # plt.ylabel("1st magnitude")
        # plt.plot(range(len(magContainer)), magContainer)
        # plt.savefig("result/layer{:02d}_firstmag_{:02d}dim_{:04d}.png".format(layerNum, compNumContainer[layerNum - 1], seed))
        # plt.show()
        # plt.clf()

    return np.array(magContainer)

def pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, layerNum, sublayerNum, interval, totalSampleNum=1000):

    magContainer = []
    print()
    print("layer {}(bb({}th))".format(layerNum, sublayerNum))
    print("mode : tube")

    for i in tqdm(range(int(totalSampleNum / interval))):
        if sublayerNum == 1:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), loadMin=interval*i + 1, loadMax=interval*(i+1))
        elif sublayerNum == 2:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)), loadMin=interval*i + 1, loadMax=interval*(i+1))
        elif sublayerNum == 3:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), loadMin=interval*i + 1, loadMax=interval*(i+1))
        
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

        # plt.title("first-magnitude of weight subspace(dim={}, layer={})".format(compNumContainer[layerNum - 1], layerNum))
        # plt.xlabel("each step")
        # plt.ylabel("1st magnitude")
        # plt.plot(range(len(magContainer)), magContainer)
        # plt.savefig("result/layer{:02d}_firstmag_{:02d}dim_{:04d}.png".format(layerNum, compNumContainer[layerNum - 1], seed))
        # plt.show()
        # plt.clf()

    return np.array(magContainer)

def pca_1stMag_layerwise_ds_CIFAR(pcaTool, smTool, seed, csvFolder, layerNum, interval, totalSampleNum=1000):

    magContainer = []
    print("layer {}(ds)".format(layerNum))

    for i in tqdm(range(int(totalSampleNum / interval))):
        _, weight_layer02 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 1)), loadMin=interval*i + 1, loadMax=interval*(i+1))
        
        dataArray1 = np.transpose(np.reshape(weight_layer02[:,0], (4*(2**layerNum), 8*(2**layerNum))))
        if i == 1:
            print(dataArray1.shape)

        dataArray2 = np.transpose(np.reshape(weight_layer02[:,1], (4*(2**layerNum), 8*(2**layerNum))))

        _, basis1 = pcaTool.pca_lowcost(dataArray1, (4*(2**layerNum)))
        _, basis2 = pcaTool.pca_lowcost(dataArray2, (4*(2**layerNum)))
        basis1 = basis1[:,:(4*(2**layerNum))]
        basis2 = basis2[:,:(4*(2**layerNum))]
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

def pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder, interval, totalSampleNum=1000):

    magContainer = []
    print("layer fc")

    for i in tqdm(range(int(totalSampleNum / interval))):
        _, weight_layerw = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer22/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
        #_, weight_layerbias = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer23/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))

        dataArray1w = np.transpose(np.reshape(weight_layerw[:,0], (10, 128)))
        #dataArray1b = np.pad(weight_layerbias[:,0][:,None], ((0,118),(0,0)), 'constant', constant_values=0)
        if i == 1:
            print(dataArray1w.shape)
            #print(dataArray1b.shape)

        #dataArray1 = np.concatenate((dataArray1w, dataArray1b), axis=1)

        dataArray2w = np.transpose(np.reshape(weight_layerw[:,1], (10, 128)))
        #dataArray2b = np.pad(weight_layerbias[:,1][:,None], ((0,118),(0,0)), 'constant', constant_values=0)

        #dataArray2 = np.concatenate((dataArray2w, dataArray2b), axis=1)

        _, basis1 = pcaTool.pca_lowcost(dataArray1w, 10)
        _, basis2 = pcaTool.pca_lowcost(dataArray2w, 10)
        basis1 = basis1[:,:10]
        basis2 = basis2[:,:10]
        if i == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
        else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

        if i == int(totalSampleNum / interval) - 1 : print(smTool.calc_magnitude(basis1, basis2))

    # plt.title("first-magnitude of weight subspace(dim=11, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_11dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def pca_2ndMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder, interval, totalSampleNum=1000):

    magContainer = []
    print("layer fc")

    for i in tqdm(range(int(totalSampleNum / interval) - 1)):
        _, weight_layerw = ma.makeDataArrayInDimSplines(glob(csvFolder + "/layer22/part*.csv"), [interval*i + 1, interval*(i + 1) + 1, interval*(i + 2)])
        #_, weight_layerbias = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer23/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))

        dataArray1w = np.transpose(np.reshape(weight_layerw[:,0], (10, 128)))
        #dataArray1b = np.pad(weight_layerbias[:,0][:,None], ((0,118),(0,0)), 'constant', constant_values=0)
        if i == 1:
            print(dataArray1w.shape)
            #print(dataArray1b.shape)

        #dataArray1 = np.concatenate((dataArray1w, dataArray1b), axis=1)

        dataArray2w = np.transpose(np.reshape(weight_layerw[:,1], (10, 128)))
        #dataArray2b = np.pad(weight_layerbias[:,1][:,None], ((0,118),(0,0)), 'constant', constant_values=0)

        #dataArray2 = np.concatenate((dataArray2w, dataArray2b), axis=1)
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

    # plt.title("first-magnitude of weight subspace(dim=11, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_11dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)


def pca_1stMag_layer2_constructionMode_CIFAR(pcaTool, smTool, seed, csvFolder, n_components, layerNum, mode):

    magContainer = []
    print("layer {}".format(layerNum))

    if mode < 5:

        for i in tqdm(range(1, 200)):
            _, weight_layer02 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 3)) + glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)) + glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 1)), loadMin=5*i-4, loadMax=5*i + 1)
            _, weight_layer03 = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)) + glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), loadMin=5*i-4, loadMax=5*i + 1)

            if mode == 1: # 존재하는 커널 8개, 4개를 1개의 단위로 벡터로 만듬(12+4+16, 1152) / 커널 1개씩 벡터로 만듬(160, 288) / 존재하는 커널 32개, 16개를 1개의 단위로 벡터로 만듬(3+1+4, 4608) / 각 층의 웨이트를 각각 벡터로 만듬(5, 9216)
                dataArray12 = np.transpose(np.reshape(weight_layer02[0:3456*(4**(layerNum-1)),0], (3*(4**(layerNum-1)), 1152)))
                dataArray13 = np.pad(np.transpose(np.reshape(weight_layer02[3456*(4**(layerNum-1)):,0], ((4**(layerNum-1), 128)))), ((0,1024),(0,0)), 'constant', constant_values=0)
                dataArray14 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),0], (4*(4**(layerNum-1)), 1152)))
                # dataArray12 = np.transpose(np.reshape(weight_layer02[0:3456*(4**(layerNum-1)),0], (3, 1152*(4**(layerNum-1)))))
                # dataArray13 = np.pad(weight_layer02[3456*(4**(layerNum-1)):,0][:,None], ((0,1024*(4**(layerNum-1))),(0,0)), 'constant', constant_values=0)
                # dataArray14 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),0], (4, 1152*(4**(layerNum-1)))))
                if i == 1:
                    print(dataArray12.shape)
                    print(dataArray13.shape)
                    print(dataArray14.shape)

                dataArray1 = np.concatenate((dataArray12, dataArray13, dataArray14), axis=1)

                dataArray22 = np.transpose(np.reshape(weight_layer02[0:3456*(4**(layerNum-1)),1], (3*(4**(layerNum-1)), 1152)))
                dataArray23 = np.pad(np.transpose(np.reshape(weight_layer02[3456*(4**(layerNum-1)):,0], ((4**(layerNum-1), 128)))), ((0,1024),(0,0)), 'constant', constant_values=0)
                dataArray24 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),1], (4*(4**(layerNum-1)), 1152)))
                # dataArray22 = np.transpose(np.reshape(weight_layer02[0:3456*(4**(layerNum-1)),1], (3, 1152*(4**(layerNum-1)))))
                # dataArray23 = np.pad(weight_layer02[3456*(4**(layerNum-1)):,1][:,None], ((0,1024*(4**(layerNum-1))),(0,0)), 'constant', constant_values=0)
                # dataArray24 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),1], (4, 1152*(4**(layerNum-1)))))

                dataArray2 = np.concatenate((dataArray22, dataArray23, dataArray24), axis=1)
            elif mode == 2:
                dataArray12 = np.pad(np.transpose(np.reshape(weight_layer02[0:4608,0], (32, 144))), ((0,144),(0,0)), 'constant', constant_values=0)
                dataArray13 = np.transpose(np.reshape(weight_layer02[4608:13824,0], (32, 288)))
                dataArray14 = np.pad(np.transpose(np.reshape(weight_layer02[13824:,0], (32, 16))), ((0,272),(0,0)), 'constant', constant_values=0)
                dataArray15 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),0], (64, 288)))
                # dataArray12 = np.transpose(np.reshape(weight_layer02[0:3456*(4**(layerNum-1)),0], (3, 1152*(4**(layerNum-1)))))
                # dataArray13 = np.pad(weight_layer02[3456*(4**(layerNum-1)):,0][:,None], ((0,1024*(4**(layerNum-1))),(0,0)), 'constant', constant_values=0)
                # dataArray14 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),0], (4, 1152*(4**(layerNum-1)))))
                if i == 1:
                    print(dataArray12.shape)
                    print(dataArray13.shape)
                    print(dataArray14.shape)
                    print(dataArray15.shape)

                dataArray1 = np.concatenate((dataArray12, dataArray13, dataArray14, dataArray15), axis=1)

                dataArray22 = np.pad(np.transpose(np.reshape(weight_layer02[0:4608,1], (32, 144))), ((0,144),(0,0)), 'constant', constant_values=0)
                dataArray23 = np.transpose(np.reshape(weight_layer02[4608:13824,1], (32, 288)))
                dataArray24 = np.pad(np.transpose(np.reshape(weight_layer02[13824:,1], (32, 16))), ((0,272),(0,0)), 'constant', constant_values=0)
                dataArray25 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),1], (64, 288)))
                # dataArray22 = np.transpose(np.reshape(weight_layer02[0:3456*(4**(layerNum-1)),1], (3, 1152*(4**(layerNum-1)))))
                # dataArray23 = np.pad(weight_layer02[3456*(4**(layerNum-1)):,1][:,None], ((0,1024*(4**(layerNum-1))),(0,0)), 'constant', constant_values=0)
                # dataArray24 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),1], (4, 1152*(4**(layerNum-1)))))

                dataArray2 = np.concatenate((dataArray22, dataArray23, dataArray24, dataArray25), axis=1)
            elif mode == 3:
                dataArray12 = weight_layer02[0:4608,0][:,None]
                dataArray13 = np.transpose(np.reshape(weight_layer02[4608:13824,0], (2, 4608)))
                dataArray14 = np.pad(weight_layer02[13824:,0][:,None], ((0,4096),(0,0)), 'constant', constant_values=0)
                dataArray15 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),0], (4, 4608)))
                # dataArray12 = np.transpose(np.reshape(weight_layer02[0:3456*(4**(layerNum-1)),0], (3, 1152*(4**(layerNum-1)))))
                # dataArray13 = np.pad(weight_layer02[3456*(4**(layerNum-1)):,0][:,None], ((0,1024*(4**(layerNum-1))),(0,0)), 'constant', constant_values=0)
                # dataArray14 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),0], (4, 1152*(4**(layerNum-1)))))
                if i == 1:
                    print(dataArray12.shape)
                    print(dataArray13.shape)
                    print(dataArray14.shape)
                    print(dataArray15.shape)

                dataArray1 = np.concatenate((dataArray12, dataArray13, dataArray14, dataArray15), axis=1)

                dataArray22 = weight_layer02[0:4608,1][:,None]
                dataArray23 = np.transpose(np.reshape(weight_layer02[4608:13824,1], (2, 4608)))
                dataArray24 = np.pad(weight_layer02[13824:,1][:,None], ((0,4096),(0,0)), 'constant', constant_values=0)
                dataArray25 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),1], (4, 4608)))
                # dataArray22 = np.transpose(np.reshape(weight_layer02[0:3456*(4**(layerNum-1)),1], (3, 1152*(4**(layerNum-1)))))
                # dataArray23 = np.pad(weight_layer02[3456*(4**(layerNum-1)):,1][:,None], ((0,1024*(4**(layerNum-1))),(0,0)), 'constant', constant_values=0)
                # dataArray24 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),1], (4, 1152*(4**(layerNum-1)))))

                dataArray2 = np.concatenate((dataArray22, dataArray23, dataArray24, dataArray25), axis=1)
            elif mode == 4:
                dataArray12 = np.pad(weight_layer02[0:4608,0][:,None], ((0,4608),(0,0)), 'constant', constant_values=0)
                dataArray13 = weight_layer02[4608:13824,0][:,None]
                dataArray14 = np.pad(weight_layer02[13824:,0][:,None], ((0,8704),(0,0)), 'constant', constant_values=0)
                dataArray15 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),0], (2, 9216)))
                # dataArray12 = np.transpose(np.reshape(weight_layer02[0:3456*(4**(layerNum-1)),0], (3, 1152*(4**(layerNum-1)))))
                # dataArray13 = np.pad(weight_layer02[3456*(4**(layerNum-1)):,0][:,None], ((0,1024*(4**(layerNum-1))),(0,0)), 'constant', constant_values=0)
                # dataArray14 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),0], (4, 1152*(4**(layerNum-1)))))
                if i == 1:
                    print(dataArray12.shape)
                    print(dataArray13.shape)
                    print(dataArray14.shape)
                    print(dataArray15.shape)

                dataArray1 = np.concatenate((dataArray12, dataArray13, dataArray14, dataArray15), axis=1)

                dataArray22 = np.pad(weight_layer02[0:4608,1][:,None], ((0,4608),(0,0)), 'constant', constant_values=0)
                dataArray23 = weight_layer02[4608:13824,1][:,None]
                dataArray24 = np.pad(weight_layer02[13824:,1][:,None], ((0,8704),(0,0)), 'constant', constant_values=0)
                dataArray25 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),1], (2, 9216)))
                # dataArray22 = np.transpose(np.reshape(weight_layer02[0:3456*(4**(layerNum-1)),1], (3, 1152*(4**(layerNum-1)))))
                # dataArray23 = np.pad(weight_layer02[3456*(4**(layerNum-1)):,1][:,None], ((0,1024*(4**(layerNum-1))),(0,0)), 'constant', constant_values=0)
                # dataArray24 = np.transpose(np.reshape(weight_layer03[0:4608*(4**(layerNum-1)),1], (4, 1152*(4**(layerNum-1)))))

                dataArray2 = np.concatenate((dataArray22, dataArray23, dataArray24, dataArray25), axis=1)

            if mode==3 or mode==4:
                lambda1, tmp1 = pcaTool.pca_lowcost(dataArray1, n_components)
                # sCon = []
                # s = 0
                # for i in range(len(lambda1)):
                #     if abs(lambda1[i]) < 0.000001: continue
                #     s += lambda1[i]
                #     sCon.append(s)
                # sVec = np.array(sCon)
                # sVec = sVec * (100 / s)
                # for i in range(sVec.shape[0]):
                #     print("{}-th step : {}".format(i, sVec[i]))
                basis1 = tmp1[:,0:n_components]
                _, tmp2 = pcaTool.pca_lowcost(dataArray2, n_components)
                basis2 = tmp2[:,0:n_components]
            else:
                lambda1, tmp1 = pcaTool.pca_basic(dataArray1)
                # sCon = []
                # s = 0
                # for i in range(len(lambda1)):
                #     if abs(lambda1[i]) < 0.000001: continue
                #     s += lambda1[i]
                #     sCon.append(s)
                # sVec = np.array(sCon)
                # sVec = sVec * (100 / s)
                # for i in range(sVec.shape[0]):
                #     print("{}-th step : {}".format(i, sVec[i]))
                basis1 = tmp1[:,0:n_components]
                _, tmp2 = pcaTool.pca_basic(dataArray2)
                basis2 = tmp2[:,0:n_components]
            # print(basis1.shape)
            # print(basis2.shape)
            if i == 1: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
            else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

    else:
        for i in tqdm(range(1, 200)):
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 3)) + glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)) + glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 1)) + glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)) + glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), loadMin=5*i-4, loadMax=5*i + 1)
            a = np.linalg.norm(weight_layer[:,0] - weight_layer[:,1])
            print(a)
            magContainer.append(a)

        # plt.title("first-magnitude of weight subspace(dim={}, layer={})".format(n_components, layerNum))
        # plt.xlabel("each step")
        # plt.ylabel("1st magnitude")
        # plt.plot(range(len(magContainer)), magContainer)
        # plt.savefig("result/layer{:02d}_firstmag_{:02d}dim_{:04d}.png".format(layerNum, n_components, seed))
        # plt.show()
        # plt.clf()

    return np.array(magContainer)

def pca_1stMag_layerwise_conv_MNIST(pcaTool, smTool, seed, csvFolder, layerNum):

    magContainer = []
    rowContainer = [9, 18, 18]
    colContainer = [2, 2, 4]

    _, dataArray_raw = ma.makeDataArray(glob(csvFolder + "/layer{}/part*.csv".format(layerNum)), 1)

    for colNum in range(int(dataArray_raw.shape[1]) - 1):
        dataArray1 = np.transpose(np.reshape(dataArray_raw[:,1], (colContainer[layerNum - 1], rowContainer[layerNum - 1])))
        dataArray2 = np.transpose(np.reshape(dataArray_raw[:,colNum + 1], (colContainer[layerNum - 1], rowContainer[layerNum - 1])))

        _, basis1 = pcaTool.pca_basic(dataArray1)
        _, basis2 = pcaTool.pca_basic(dataArray2)
        basis1 = basis1[:,:colContainer[layerNum - 1]]
        basis2 = basis2[:,:colContainer[layerNum - 1]]
        if colNum == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
        else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

        if colNum == int(dataArray_raw.shape[1]) - 2 : print(smTool.calc_magnitude(basis1, basis2))

    return np.array(magContainer)

    plt.title("first-magnitude of weight subspace(dim=7, layer=all)")
    plt.xlabel("each step")
    plt.ylabel("1st magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/layer_firstmag_7dim_MNIST_{:04d}.png".format(seed))
    plt.show()

def pca_1stMag_tube_conv_MNIST(pcaTool, smTool, seed, csvFolder, layerNum):

    magContainer = []
    colContainer = [2, 4, 8]

    _, dataArray_raw = ma.makeDataArray(glob(csvFolder + "/layer{}/part*.csv".format(layerNum)), 1)

    for colNum in range(int(dataArray_raw.shape[1]) - 1):
        dataArray1 = np.reshape(dataArray_raw[:,colNum], (colContainer[layerNum - 1], 9))
        dataArray2 = np.reshape(dataArray_raw[:,colNum + 1], (colContainer[layerNum - 1], 9))

        _, basis1 = pcaTool.pca_basic(dataArray1)
        _, basis2 = pcaTool.pca_basic(dataArray2)
        basis1 = basis1[:,:colContainer[layerNum - 1]]
        basis2 = basis2[:,:colContainer[layerNum - 1]]
        if colNum == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
        else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

        if colNum == int(dataArray_raw.shape[1]) - 2 : print(smTool.calc_magnitude(basis1, basis2))

    return np.array(magContainer)

    plt.title("first-magnitude of weight subspace(dim=7, layer=all)")
    plt.xlabel("each step")
    plt.ylabel("1st magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/layer_firstmag_7dim_MNIST_{:04d}.png".format(seed))
    plt.show()

def pca_1stMag_layerwise_fc_MNIST(pcaTool, smTool, seed, csvFolder):

    magContainer = []

    _, dataArray_raw4 = ma.makeDataArray(glob(csvFolder + "/layer4/part*.csv"), 1)

    for colNum in range(int(dataArray_raw4.shape[1]) - 1):
        dataArray1 = np.transpose(np.reshape(dataArray_raw4[:,1], (10, 36)))
        dataArray2 = np.transpose(np.reshape(dataArray_raw4[:,colNum + 1], (10, 36)))

        _, basis1 = pcaTool.pca_basic(dataArray1)
        _, basis2 = pcaTool.pca_basic(dataArray2)
        basis1 = basis1[:,:10]
        basis2 = basis2[:,:10]
        if colNum == 0: magContainer.append(smTool.calc_magnitude(basis1, basis2, tmp=True))
        else : magContainer.append(smTool.calc_magnitude(basis1, basis2))

        if colNum == int(dataArray_raw4.shape[1]) - 2 : print(smTool.calc_magnitude(basis1, basis2))

    return np.array(magContainer)

    plt.title("first-magnitude of weight subspace(dim=10, layer=fc)")
    plt.xlabel("each step")
    plt.ylabel("1st magnitude")
    plt.plot(range(len(magContainer)), magContainer)
    plt.savefig("result/layer_firstmag_10dim_MNIST_{:04d}.png".format(seed))
    plt.show()


# ------------------------------------------------------------------

dataset = "CIFAR"

if dataset == "MNIST":
    
    pcaTool = wp.weightPCA()
    smTool = sm.SubspaceDiff()

    seed = 8481
    
    csvFolderList = glob("C:/Users/dmtsa/research/run_analysis/DB_CNN_MNIST_1/1-4-8-fc__*")
    csvFolder = "C:/Users/dmtsa/research/run_analysis/DB_CNN_MNIST_3/1-2-2-4-fc__{:04d}".format(seed)

    conv1 = pca_1stMag_layerwise_conv_MNIST(pcaTool, smTool, seed, csvFolder, 1) /1.33
    conv2 = pca_1stMag_layerwise_conv_MNIST(pcaTool, smTool, seed, csvFolder, 2) /2.61
    conv3 = pca_1stMag_layerwise_conv_MNIST(pcaTool, smTool, seed, csvFolder, 3) /3.77
    fc = pca_1stMag_layerwise_fc_MNIST(pcaTool, smTool, seed, csvFolder) /9.61

    window = 10 # 移動平均の範囲
    w = np.ones(window)/window

    conv1 = np.convolve(np.concatenate((conv1, [1]*10)), w, mode='same')[:200]
    conv2 = np.convolve(np.concatenate((conv2, [1]*10)), w, mode='same')[:200]
    conv3 = np.convolve(np.concatenate((conv3, [1]*10)), w, mode='same')[:200]
    fc = np.convolve(np.concatenate((fc, [1]*10)), w, mode='same')[:200]

    f = open(csvFolder + "/accContainer.csv", "r")
    accRaw = f.readline().split(",")
    accContainer = []
    for i, text in enumerate(accRaw):
        if i%10 == 0: accContainer.append(float(text))
        # accContainer.append(float(text))

    plt.title("magnitude of weight subspace(dim=2, 2, 4, 10)")
    plt.xlabel("each step")
    plt.ylabel("magnitude / accuracy")
    plt.grid()
    plt.plot(range(len(conv1)), conv1, label="con layer 1")
    plt.plot(range(len(conv2)), conv2, label="con layer 2")
    plt.plot(range(len(conv3)), conv3, label="con layer 3")
    plt.plot(range(len(fc)), fc, label="fc layer")
    #plt.plot(range(len(accContainer)), accContainer, label="accuracy[scale=1]")
    plt.legend()
    plt.savefig("result/total_firstmag_2_2_4_10dim_MNIST_{:04d}.png".format(seed))
    # plt.show()

    # plt.rc('xtick', labelsize=5)
    # aa = [7.07, 5.56, 4.86, 4.21, 11.86, 10.92, 9.8, 9.06, 25.73, 22.34, 21.07, 16.8, 46.24, 20.56, 11.58, 6.26]
    # x = ["block 1\n(input)", "block 1\n(first)", "block 1\n(second)", "block 1\n(third)", "block 2\n(input)", "block 2\n(first)", "block 2\n(second)", "block 2\n(third)", "block 3\n(input)", "block 3\n(first)", "block 3\n(second)", "block 3\n(third)", "block 4\n(input)", "block 4\n(first)", "block 4\n(second)", "block 4\n(third)"]
    # plt.title("final cumulative-magnitude of weight subspace")
    # plt.xlabel("each layer")
    # plt.ylabel("cumulative magnitude")
    # plt.grid()
    # plt.plot(x, aa)
    # plt.savefig("result/total_firstmag_final_culmulative_CIFAR_{:04d}.png".format(1827))

elif dataset == "CIFAR":

    pcaTool = wp.weightPCA()
    smTool = sm.SubspaceDiff()

    seed = 1827
    interval = 10
    upperFlag = False
    secondFlag = True
    mode = "graph"
    layer = 2

    csvFolderList = glob("C:/Users/dmtsa/research/run_analysis/DB_CNN_MNIST_1/1-4-8-fc__*")
    csvFolder = "C:/Users/dmtsa/research/run_analysis/DB_RESNET18_CIFAR_8/resnet18__{:04d}".format(seed)

    # # f = open(csvFolder + "/accContainer.csv", "r")
    # # accRaw = f.readline().split(",")
    # # accContainer = []
    # # for i, text in enumerate(accRaw):
    # #     if i%25 == 0: accContainer.append(float(text) / 2)
    # # f.close()

    if mode == "graph":
        if upperFlag:
            if secondFlag:
                if layer == 0:
                    i1 = pca_2ndMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder, interval)
                    f1 = pca_2ndMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder, interval)

                    plt.title("second-magnitude of weight subspace(layer : input, fc)")
                    plt.xlabel("each epoch")
                    plt.ylabel("2nd magnitude")
                    plt.plot(range(len(i1)), i1, label="layer input")
                    plt.plot(range(len(f1)), f1, label="layer fc")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/ipfc_secondmag_interval{}_8_10dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 1:
                    n1 = pca_2ndMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 1, interval)
                    n2 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 1, interval)
                    n3 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 2, interval)
                    n4 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 3, interval)

                    plt.title("second-magnitude of weight subspace(each layer in basicblock1)")
                    plt.xlabel("each epoch")
                    plt.ylabel("2nd magnitude")
                    plt.plot(range(len(n1)), n1, label="block 1(bb input)")
                    plt.plot(range(len(n2)), n2, label="block 1(bb first)")
                    plt.plot(range(len(n3)), n3, label="block 1(bb second)")
                    plt.plot(range(len(n4)), n4, label="block 1(bb third)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/b1_i123_secondmag_interval{}_all16dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 2:
                    n1 = pca_2ndMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 2, interval)
                    n2 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 1, interval)
                    n3 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 2, interval)
                    n4 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 3, interval)

                    plt.title("second-magnitude of weight subspace(each layer in basicblock2)")
                    plt.xlabel("each epoch")
                    plt.ylabel("2nd magnitude")
                    plt.plot(range(len(n1)), n1, label="block 2(bb input)")
                    plt.plot(range(len(n2)), n2, label="block 2(bb first)")
                    plt.plot(range(len(n3)), n3, label="block 2(bb second)")
                    plt.plot(range(len(n4)), n4, label="block 2(bb third)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/b2_i123_secondmag_interval{}_all32dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 3:
                    n1 = pca_2ndMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 3, interval)
                    n2 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 1, interval)
                    n3 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 2, interval)
                    n4 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 3, interval)

                    plt.title("second-magnitude of weight subspace(each layer in basicblock3)")
                    plt.xlabel("each epoch")
                    plt.ylabel("2nd magnitude")
                    plt.plot(range(len(n1)), n1, label="block 3(bb input)")
                    plt.plot(range(len(n2)), n2, label="block 3(bb first)")
                    plt.plot(range(len(n3)), n3, label="block 3(bb second)")
                    plt.plot(range(len(n4)), n4, label="block 3(bb third)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/b3_i123_secondmag_interval{}_all64dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 4:
                    n1 = pca_2ndMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 4, interval)
                    n2 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 1, interval)
                    n3 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 2, interval)
                    n4 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 3, interval)

                    plt.title("second-magnitude of weight subspace(each layer in basicblock4)")
                    plt.xlabel("each epoch")
                    plt.ylabel("2nd magnitude")
                    plt.plot(range(len(n1)), n1, label="block 4(bb input)")
                    plt.plot(range(len(n2)), n2, label="block 4(bb first)")
                    plt.plot(range(len(n3)), n3, label="block 4(bb second)")
                    plt.plot(range(len(n4)), n4, label="block 4(bb third)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/b4_i123_secondmag_interval{}_all128dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
            else:
                if layer == 0:
                    i1 = pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder, interval)
                    f1 = pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder, interval)

                    plt.title("first-magnitude of weight subspace(layer : input, fc)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st magnitude")
                    plt.plot(range(len(i1)), i1, label="layer input")
                    plt.plot(range(len(f1)), f1, label="layer fc")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/ipfc_firstmag_interval{}_8_10dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 1:
                    n1 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 1, interval)
                    n2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 1, interval)
                    n3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 2, interval)
                    n4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 3, interval)

                    plt.title("first-magnitude of weight subspace(each layer in basicblock1)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st magnitude")
                    plt.plot(range(len(n1)), n1, label="block 1(bb input)")
                    plt.plot(range(len(n2)), n2, label="block 1(bb first)")
                    plt.plot(range(len(n3)), n3, label="block 1(bb second)")
                    plt.plot(range(len(n4)), n4, label="block 1(bb third)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/b1_i123_firstmag_interval{}_all16dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 2:
                    n1 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 2, interval)
                    n2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 1, interval)
                    n3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 2, interval)
                    n4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 3, interval)

                    plt.title("first-magnitude of weight subspace(each layer in basicblock2)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st magnitude")
                    plt.plot(range(len(n1)), n1, label="block 2(bb input)")
                    plt.plot(range(len(n2)), n2, label="block 2(bb first)")
                    plt.plot(range(len(n3)), n3, label="block 2(bb second)")
                    plt.plot(range(len(n4)), n4, label="block 2(bb third)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/b2_i123_firstmag_interval{}_all32dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 3:
                    n1 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 3, interval)
                    n2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 1, interval)
                    n3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 2, interval)
                    n4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 3, interval)

                    plt.title("first-magnitude of weight subspace(each layer in basicblock3)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st magnitude")
                    plt.plot(range(len(n1)), n1, label="block 3(bb input)")
                    plt.plot(range(len(n2)), n2, label="block 3(bb first)")
                    plt.plot(range(len(n3)), n3, label="block 3(bb second)")
                    plt.plot(range(len(n4)), n4, label="block 3(bb third)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/b3_i123_firstmag_interval{}_all64dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 4:
                    n1 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 4, interval)
                    n2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 1, interval)
                    n3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 2, interval)
                    n4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 3, interval)

                    plt.title("first-magnitude of weight subspace(each layer in basicblock4)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st magnitude")
                    plt.plot(range(len(n1)), n1, label="block 4(bb input)")
                    plt.plot(range(len(n2)), n2, label="block 4(bb first)")
                    plt.plot(range(len(n3)), n3, label="block 4(bb second)")
                    plt.plot(range(len(n4)), n4, label="block 4(bb third)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/b4_i123_firstmag_interval{}_all128dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
        else:
            if secondFlag:
                if layer == 0:
                    ii1 = pca_2ndMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder, interval)
                    ff1 = pca_2ndMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder, interval)

                    i1 = pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder, interval)
                    f1 = pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder, interval)

                    plt.title("magnitude of weight subspace(layer : input, fc)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st, 2nd magnitude")
                    plt.plot(range(len(i1)), i1, label="layer input(1st mag)")
                    plt.plot(range(len(f1)), f1, label="layer fc(1st mag)")
                    plt.plot(range(1, len(ii1)+1), ii1, label="layer input(2nd mag)")
                    plt.plot(range(1, len(ff1)+1), ff1, label="layer fc(2nd mag)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/ipfc_mag_interval{}_8_10dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 1:
                    nn1 = pca_2ndMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 1, interval)
                    nn2 = pca_2ndMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 2, interval)
                    nn3 = pca_2ndMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 3, interval)
                    nn4 = pca_2ndMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 4, interval)

                    n1 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 1, interval) #/7.07
                    n2 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 2, interval) #/11.86
                    n3 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 3, interval) #/25.73
                    n4 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 4, interval) #/46.24

                    plt.title("magnitude of weight subspace(sub-input layer)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st, 2nd magnitude")
                    plt.plot(range(len(n1)), n1, label="block 1(bb input, (1st mag))")
                    plt.plot(range(len(n2)), n2, label="block 2(bb input, (1st mag))")
                    plt.plot(range(len(n3)), n3, label="block 3(bb input, (1st mag))")
                    plt.plot(range(len(n4)), n4, label="block 4(bb input, (1st mag))")
                    plt.plot(range(1, len(nn1)+1), nn1, label="block 1(bb input, (2nd mag))")
                    plt.plot(range(1, len(nn2)+1), nn2, label="block 2(bb input, (2nd mag))")
                    plt.plot(range(1, len(nn3)+1), nn3, label="block 3(bb input, (2nd mag))")
                    plt.plot(range(1, len(nn4)+1), nn4, label="block 4(bb input, (2nd mag))")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/bbi_mag_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 2:
                    cc11 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 1, interval)
                    cc12 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 1, interval)
                    cc13 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 1, interval)
                    cc14 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 1, interval)

                    c11 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 1, interval) #/5.56
                    c12 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 1, interval) #/10.92
                    c13 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 1, interval) #/22.34
                    c14 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 1, interval) #/20.56

                    plt.title("magnitude of weight subspace(sub-first layer)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st, 2nd magnitude")
                    plt.plot(range(len(c11)), c11, label="block 1(bb first, (1st mag))")
                    plt.plot(range(len(c12)), c12, label="block 2(bb first, (1st mag))")
                    plt.plot(range(len(c13)), c13, label="block 3(bb first, (1st mag))")
                    plt.plot(range(len(c14)), c14, label="block 4(bb first, (1st mag))")
                    plt.plot(range(1, len(cc11)+1), cc11, label="block 1(bb first, (2nd mag))")
                    plt.plot(range(1, len(cc12)+1), cc12, label="block 2(bb first, (2nd mag))")
                    plt.plot(range(1, len(cc13)+1), cc13, label="block 3(bb first, (2nd mag))")
                    plt.plot(range(1, len(cc14)+1), cc14, label="block 4(bb first, (2nd mag))")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/bb1_mag_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 3:
                    cc21 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 2, interval)
                    cc22 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 2, interval)
                    cc23 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 2, interval)
                    cc24 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 2, interval)

                    c21 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 2, interval) #/4.86
                    c22 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 2, interval) #/9.8
                    c23 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 2, interval) #/21.07
                    c24 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 2, interval) #/11.58

                    plt.title("magnitude of weight subspace(sub-second layer)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st, 2nd magnitude")
                    plt.plot(range(len(c21)), c21, label="block 1(bb second, (1st mag))")
                    plt.plot(range(len(c22)), c22, label="block 2(bb second, (1st mag))")
                    plt.plot(range(len(c23)), c23, label="block 3(bb second, (1st mag))")
                    plt.plot(range(len(c24)), c24, label="block 4(bb second, (1st mag))")
                    plt.plot(range(1, len(cc21)+1), cc21, label="block 1(bb second, (2nd mag))")
                    plt.plot(range(1, len(cc22)+1), cc22, label="block 2(bb second, (2nd mag))")
                    plt.plot(range(1, len(cc23)+1), cc23, label="block 3(bb second, (2nd mag))")
                    plt.plot(range(1, len(cc24)+1), cc24, label="block 4(bb second, (2nd mag))")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/bb2_mag_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 4:
                    cc31 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 3, interval)
                    cc32 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 3, interval)
                    cc33 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 3, interval)
                    cc34 = pca_2ndMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 3, interval)

                    c31 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 3, interval) #/4.21
                    c32 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 3, interval) #/9.06
                    c33 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 3, interval) #/16.8
                    c34 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 3, interval) #/6.26

                    plt.title("magnitude of weight subspace(sub-third layer)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st, 2nd magnitude")
                    plt.plot(range(len(c31)), c31, label="block 1(bb third, (1st mag))")
                    plt.plot(range(len(c32)), c32, label="block 2(bb third, (1st mag))")
                    plt.plot(range(len(c33)), c33, label="block 3(bb third, (1st mag))")
                    plt.plot(range(len(c34)), c34, label="block 4(bb third, (1st mag))")
                    plt.plot(range(1, len(cc31)+1), cc31, label="block 1(bb third, (2nd mag))")
                    plt.plot(range(1, len(cc32)+1), cc32, label="block 2(bb third, (2nd mag))")
                    plt.plot(range(1, len(cc33)+1), cc33, label="block 3(bb third, (2nd mag))")
                    plt.plot(range(1, len(cc34)+1), cc34, label="block 4(bb third, (2nd mag))")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/bb3_mag_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
            else:
                if layer == 0:
                    i1 = pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder, interval)
                    f1 = pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder, interval)

                    plt.title("first-magnitude of weight subspace(layer : input, fc)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st magnitude")
                    plt.plot(range(len(i1)), i1, label="layer input")
                    plt.plot(range(len(f1)), f1, label="layer fc")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/ipfc_firstmag_interval{}_8_10dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 1:
                    n1 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 1, interval) #/7.07
                    n2 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 2, interval) #/11.86
                    n3 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 3, interval) #/25.73
                    n4 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 4, interval) #/46.24

                    plt.title("first-magnitude of weight subspace(sub-input layer)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st magnitude")
                    plt.plot(range(len(n1)), n1, label="block 1(bb input)")
                    plt.plot(range(len(n2)), n2, label="block 2(bb input)")
                    plt.plot(range(len(n3)), n3, label="block 3(bb input)")
                    plt.plot(range(len(n4)), n4, label="block 4(bb input)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/bbi_firstmag_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 2:
                    c11 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 1, interval) #/5.56
                    c12 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 1, interval) #/10.92
                    c13 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 1, interval) #/22.34
                    c14 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 1, interval) #/20.56

                    plt.title("first-magnitude of weight subspace(sub-first layer)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st magnitude")
                    plt.plot(range(len(c11)), c11, label="block 1(bb first)")
                    plt.plot(range(len(c12)), c12, label="block 2(bb first)")
                    plt.plot(range(len(c13)), c13, label="block 3(bb first)")
                    plt.plot(range(len(c14)), c14, label="block 4(bb first)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/bb1_firstmag_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 3:
                    c21 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 2, interval) #/4.86
                    c22 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 2, interval) #/9.8
                    c23 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 2, interval) #/21.07
                    c24 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 2, interval) #/11.58

                    plt.title("first-magnitude of weight subspace(sub-second layer)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st magnitude")
                    plt.plot(range(len(c21)), c21, label="block 1(bb second)")
                    plt.plot(range(len(c22)), c22, label="block 2(bb second)")
                    plt.plot(range(len(c23)), c23, label="block 3(bb second)")
                    plt.plot(range(len(c24)), c24, label="block 4(bb second)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/bb2_firstmag_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
                elif layer == 4:
                    c31 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 3, interval) #/4.21
                    c32 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 3, interval) #/9.06
                    c33 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 3, interval) #/16.8
                    c34 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 3, interval) #/6.26

                    plt.title("first-magnitude of weight subspace(sub-third layer)")
                    plt.xlabel("each epoch")
                    plt.ylabel("1st magnitude")
                    plt.plot(range(len(c31)), c31, label="block 1(bb third)")
                    plt.plot(range(len(c32)), c32, label="block 2(bb third)")
                    plt.plot(range(len(c33)), c33, label="block 3(bb third)")
                    plt.plot(range(len(c34)), c34, label="block 4(bb third)")
                    plt.grid()
                    plt.legend()
                    plt.savefig("result/bb3_firstmag_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                    # plt.show()
                    plt.clf()
    elif mode == "tube":
        if upperFlag:
            if layer == 0:
                i1 = pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder, interval)
                f1 = pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder, interval)

                plt.title("first-magnitude of weight subspace(layer : input, fc)")
                plt.xlabel("each epoch")
                plt.ylabel("1st magnitude")
                plt.plot(range(len(i1)), i1, label="layer input")
                plt.plot(range(len(f1)), f1, label="layer fc")
                plt.grid()
                plt.legend()
                plt.savefig("result/ipfc_firstmag_frominit_interval{}_8_10dim_{:04d}.png".format(interval, seed))
                # plt.show()
                plt.clf()
            elif layer == 1:
                n1 = pca_1stMag_tube_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 1, interval)
                n2 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 1, interval)
                n3 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 2, interval)
                n4 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 3, interval)

                plt.title("first-magnitude of weight subspace(each layer in basicblock1)")
                plt.xlabel("each epoch")
                plt.ylabel("1st magnitude")
                plt.plot(range(len(n1)), n1, label="block 1(bb input)")
                plt.plot(range(len(n2)), n2, label="block 1(bb first)")
                plt.plot(range(len(n3)), n3, label="block 1(bb second)")
                plt.plot(range(len(n4)), n4, label="block 1(bb third)")
                plt.grid()
                plt.legend()
                plt.savefig("result/b1_i123_firstmag_frominit_tube_interval{}_all16dim_{:04d}.png".format(interval, seed))
                # plt.show()
                plt.clf()
            elif layer == 2:
                n1 = pca_1stMag_tube_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 2, interval)
                n2 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 1, interval)
                n3 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 2, interval)
                n4 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 3, interval)

                plt.title("first-magnitude of weight subspace(each layer in basicblock2)")
                plt.xlabel("each epoch")
                plt.ylabel("1st magnitude")
                plt.plot(range(len(n1)), n1, label="block 2(bb input)")
                plt.plot(range(len(n2)), n2, label="block 2(bb first)")
                plt.plot(range(len(n3)), n3, label="block 2(bb second)")
                plt.plot(range(len(n4)), n4, label="block 2(bb third)")
                plt.grid()
                plt.legend()
                plt.savefig("result/b2_i123_firstmag_frominit_tube_interval{}_all32dim_{:04d}.png".format(interval, seed))
                # plt.show()
                plt.clf()
            elif layer == 3:
                n1 = pca_1stMag_tube_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 3, interval)
                n2 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 1, interval)
                n3 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 2, interval)
                n4 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 3, interval)

                plt.title("first-magnitude of weight subspace(each layer in basicblock3)")
                plt.xlabel("each epoch")
                plt.ylabel("1st magnitude")
                plt.plot(range(len(n1)), n1, label="block 3(bb input)")
                plt.plot(range(len(n2)), n2, label="block 3(bb first)")
                plt.plot(range(len(n3)), n3, label="block 3(bb second)")
                plt.plot(range(len(n4)), n4, label="block 3(bb third)")
                plt.grid()
                plt.legend()
                plt.savefig("result/b3_i123_firstmag_frominit_tube_interval{}_all64dim_{:04d}.png".format(interval, seed))
                # plt.show()
                plt.clf()
            elif layer == 4:
                n1 = pca_1stMag_tube_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 4, interval)
                n2 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 1, interval)
                n3 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 2, interval)
                n4 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 3, interval)

                plt.title("first-magnitude of weight subspace(each layer in basicblock4)")
                plt.xlabel("each epoch")
                plt.ylabel("1st magnitude")
                plt.plot(range(len(n1)), n1, label="block 4(bb input)")
                plt.plot(range(len(n2)), n2, label="block 4(bb first)")
                plt.plot(range(len(n3)), n3, label="block 4(bb second)")
                plt.plot(range(len(n4)), n4, label="block 4(bb third)")
                plt.grid()
                plt.legend()
                plt.savefig("result/b4_i123_firstmag_frominit_tube_interval{}_all128dim_{:04d}.png".format(interval, seed))
                # plt.show()
                plt.clf()
        else:
            if layer == 0:
                i1 = pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder, interval)
                f1 = pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder, interval)

                plt.title("first-magnitude of weight subspace(layer : input, fc)")
                plt.xlabel("each epoch")
                plt.ylabel("1st magnitude")
                plt.plot(range(len(i1)), i1, label="layer input")
                plt.plot(range(len(f1)), f1, label="layer fc")
                plt.grid()
                plt.legend()
                plt.savefig("result/ipfc_firstmag_frominit_interval{}_8_10dim_{:04d}.png".format(interval, seed))
                # plt.show()
                plt.clf()
            elif layer == 1:
                n1 = pca_1stMag_tube_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 1, interval)
                n2 = pca_1stMag_tube_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 2, interval)
                n3 = pca_1stMag_tube_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 3, interval)
                n4 = pca_1stMag_tube_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 4, interval)

                plt.title("first-magnitude of weight subspace(layer : sub-input layer)")
                plt.xlabel("each epoch")
                plt.ylabel("1st magnitude")
                plt.plot(range(len(n1)), n1, label="block 1(bb input)")
                plt.plot(range(len(n2)), n2, label="block 2(bb input)")
                plt.plot(range(len(n3)), n3, label="block 3(bb input)")
                plt.plot(range(len(n4)), n4, label="block 4(bb input)")
                plt.grid()
                plt.legend()
                plt.savefig("result/bbi_firstmag_frominit_tube_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                # plt.show()
                plt.clf()
            elif layer == 2:
                c11 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 1, interval)
                c12 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 1, interval)
                c13 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 1, interval)
                c14 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 1, interval)

                plt.title("first-magnitude of weight subspace(layer : sub-first layer)")
                plt.xlabel("each epoch")
                plt.ylabel("1st magnitude")
                plt.plot(range(len(c11)), c11, label="block 1(bb first)")
                plt.plot(range(len(c12)), c12, label="block 2(bb first)")
                plt.plot(range(len(c13)), c13, label="block 3(bb first)")
                plt.plot(range(len(c14)), c14, label="block 4(bb first)")
                plt.grid()
                plt.legend()
                plt.savefig("result/bb1_firstmag_frominit_tube_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                # plt.show()
                plt.clf()
            elif layer == 3:
                c21 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 2, interval)
                c22 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 2, interval)
                c23 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 2, interval)
                c24 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 2, interval)

                plt.title("first-magnitude of weight subspace(layer : sub-second layer)")
                plt.xlabel("each epoch")
                plt.ylabel("1st magnitude")
                plt.plot(range(len(c21)), c21, label="block 1(bb second)")
                plt.plot(range(len(c22)), c22, label="block 2(bb second)")
                plt.plot(range(len(c23)), c23, label="block 3(bb second)")
                plt.plot(range(len(c24)), c24, label="block 4(bb second)")
                plt.grid()
                plt.legend()
                plt.savefig("result/bb2_firstmag_frominit_tube_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                # plt.show()
                plt.clf()
            elif layer == 4:
                c31 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 3, interval)
                c32 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 3, interval)
                c33 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 3, interval)
                c34 = pca_1stMag_tube_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 3, interval)

                plt.title("first-magnitude of weight subspace(layer : sub-third layer)")
                plt.xlabel("each epoch")
                plt.ylabel("1st magnitude")
                plt.plot(range(len(c31)), c31, label="block 1(bb third)")
                plt.plot(range(len(c32)), c32, label="block 2(bb third)")
                plt.plot(range(len(c33)), c33, label="block 3(bb third)")
                plt.plot(range(len(c34)), c34, label="block 4(bb third)")
                plt.grid()
                plt.legend()
                plt.savefig("result/bb3_firstmag_frominit_tube_interval{}_16_32_64_128dim_{:04d}.png".format(interval, seed))
                # plt.show()
                plt.clf()
    elif mode == "animation" :

        a1 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 1, interval)
        a2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 1, interval)
        a3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 2, interval)
        a4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1, 3, interval)

        b1 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 2, interval)
        b2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 1, interval)
        b3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 2, interval)
        b4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2, 3, interval)

        c1 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 3, interval)
        c2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 1, interval)
        c3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 2, interval)
        c4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3, 3, interval)

        d1 = pca_1stMag_layerwise_bbi_CIFAR(pcaTool, smTool, seed, csvFolder, 4, interval)
        d2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 1, interval)
        d3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 2, interval)
        d4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4, 3, interval)

        
        frame = []
        for i in range(len(a1)):
            tmp = []
            tmp.append(a1[i])
            tmp.append(a2[i])
            tmp.append(a3[i])
            tmp.append(a4[i])
            tmp.append(b1[i])
            tmp.append(b2[i])
            tmp.append(b3[i])
            tmp.append(b4[i])
            tmp.append(c1[i])
            tmp.append(c2[i])
            tmp.append(c3[i])
            tmp.append(c4[i])
            tmp.append(d1[i])
            tmp.append(d2[i])
            tmp.append(d3[i])
            tmp.append(d4[i])
            frame.append(tmp)
        
        fig, ax = plt.subplots()
        ax.set_xlim(0, 17)
        ax.set_ylim(0, 0.6)
        
        x = np.arange(1, 17)
        y = []
        line, = plt.plot([], [], 'b')

        plt.title("first-magnitude of each layer's weight subspace")
        plt.xlabel("each layer")
        plt.ylabel("1st magnitude")
        plt.grid()

        def update(frame):
            y = frame
            line.set_data(x, y)
            return line,

        anim = FuncAnimation(fig, update, frames=frame)
        anim.save('depth_firstmag_{:04d}.gif'.format(seed), writer='imagemagick')

        # for frameNum in range(10):
        #     plt.title("first-magnitude of each layer's weight subspace")
        #     plt.xlabel("each layer")
        #     plt.ylabel("1st magnitude")
        #     plt.plot(np.arange(1, 17), frame[frameNum])
        #     plt.grid()
        #     plt.savefig("result/depth_firstmag_frame{:02d}_{:04d}.png".format(frameNum, seed))
        #     # plt.show()
        #     plt.clf()


    # seed = 114

    # csvFolderList = glob("C:/Users/dmtsa/research/run_analysis/DB_CNN_MNIST_1/1-4-8-fc__*")
    # csvFolder = "C:/Users/dmtsa/research/run_analysis/DB_RESNET18_CIFAR_5/resnet18__{:04d}".format(seed)

    # l0 = pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder)
    # l1 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1)
    # l2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2)
    # l3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3)
    # l4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4)
    # lf = pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder)

    # f = open(csvFolder + "/accContainer.csv", "r")
    # accRaw = f.readline().split(",")
    # accContainer = []
    # for i, text in enumerate(accRaw):
    #     accContainer.append(float(text) / 2)
    # f.close()

    # plt.title("first-magnitude of weight subspace(dim=24, 8, 32, 128, 512, 11)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(l0)), l0, label="block input")
    # plt.plot(range(len(l1)), l1, label="block 1")
    # plt.plot(range(len(l2)), l2, label="block 2")
    # plt.plot(range(len(l3)), l3, label="block 3")
    # plt.plot(range(len(l4)), l4, label="block 4")
    # plt.plot(range(len(lf)), lf, label="fc layer")
    # plt.plot(range(len(accContainer)), accContainer, label="accuracy[scale=0.5]")
    # plt.legend()
    # plt.savefig("result/total_firstmag_24_8_32_128_512_11dim_{:04d}.png".format(seed))
    # # plt.show()
    # plt.clf()

    # seed = 4646

    # csvFolderList = glob("C:/Users/dmtsa/research/run_analysis/DB_CNN_MNIST_1/1-4-8-fc__*")
    # csvFolder = "C:/Users/dmtsa/research/run_analysis/DB_RESNET18_CIFAR_5/resnet18__{:04d}".format(seed)

    # l0 = pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder)
    # l1 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1)
    # l2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2)
    # l3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3)
    # l4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4)
    # lf = pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder)

    # f = open(csvFolder + "/accContainer.csv", "r")
    # accRaw = f.readline().split(",")
    # accContainer = []
    # for i, text in enumerate(accRaw):
    #     accContainer.append(float(text) / 2)
    # f.close()

    # plt.title("first-magnitude of weight subspace(dim=24, 8, 32, 128, 512, 11)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(l0)), l0, label="block input")
    # plt.plot(range(len(l1)), l1, label="block 1")
    # plt.plot(range(len(l2)), l2, label="block 2")
    # plt.plot(range(len(l3)), l3, label="block 3")
    # plt.plot(range(len(l4)), l4, label="block 4")
    # plt.plot(range(len(lf)), lf, label="fc layer")
    # plt.plot(range(len(accContainer)), accContainer, label="accuracy[scale=0.5]")
    # plt.legend()
    # plt.savefig("result/total_firstmag_24_8_32_128_512_11dim_{:04d}.png".format(seed))
    # # plt.show()
    # plt.clf()

    # seed = 6914

    # csvFolderList = glob("C:/Users/dmtsa/research/run_analysis/DB_CNN_MNIST_1/1-4-8-fc__*")
    # csvFolder = "C:/Users/dmtsa/research/run_analysis/DB_RESNET18_CIFAR_5/resnet18__{:04d}".format(seed)

    # l0 = pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder)
    # l1 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1)
    # l2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2)
    # l3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3)
    # l4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4)
    # lf = pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder)

    # f = open(csvFolder + "/accContainer.csv", "r")
    # accRaw = f.readline().split(",")
    # accContainer = []
    # for i, text in enumerate(accRaw):
    #     if i%5 ==0:
    #         accContainer.append(float(text) / 2)
    # f.close()

    # plt.title("first-magnitude of weight subspace(dim=24, 8, 32, 128, 512, 11)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(l0)), l0, label="block input")
    # plt.plot(range(len(l1)), l1, label="block 1")
    # plt.plot(range(len(l2)), l2, label="block 2")
    # plt.plot(range(len(l3)), l3, label="block 3")
    # plt.plot(range(len(l4)), l4, label="block 4")
    # plt.plot(range(len(lf)), lf, label="fc layer")
    # plt.plot(range(len(accContainer)), accContainer, label="accuracy[scale=0.5]")
    # plt.legend()
    # plt.savefig("result/total_firstmag_24_8_32_128_512_11dim_{:04d}.png".format(seed))
    # # plt.show()
    # plt.clf()

    # seed = 7659

    # csvFolderList = glob("C:/Users/dmtsa/research/run_analysis/DB_CNN_MNIST_1/1-4-8-fc__*")
    # csvFolder = "C:/Users/dmtsa/research/run_analysis/DB_RESNET18_CIFAR_5/resnet18__{:04d}".format(seed)

    # l0 = pca_1stMag_layerwise_ip_CIFAR(pcaTool, smTool, seed, csvFolder)
    # l1 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 1)
    # l2 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 2)
    # l3 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 3)
    # l4 = pca_1stMag_layerwise_bb_CIFAR(pcaTool, smTool, seed, csvFolder, 4)
    # lf = pca_1stMag_layerwise_fc_CIFAR(pcaTool, smTool, seed, csvFolder)

    # f = open(csvFolder + "/accContainer.csv", "r")
    # accRaw = f.readline().split(",")
    # accContainer = []
    # for i, text in enumerate(accRaw):
    #     if i%5 ==0:
    #         accContainer.append(float(text) / 2)
    # f.close()

    # plt.title("first-magnitude of weight subspace(dim=24, 8, 32, 128, 512, 11)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(l0)), l0, label="block input")
    # plt.plot(range(len(l1)), l1, label="block 1")
    # plt.plot(range(len(l2)), l2, label="block 2")
    # plt.plot(range(len(l3)), l3, label="block 3")
    # plt.plot(range(len(l4)), l4, label="block 4")
    # plt.plot(range(len(lf)), lf, label="fc layer")
    # plt.plot(range(len(accContainer)), accContainer, label="accuracy[scale=0.5]")
    # plt.legend()
    # plt.savefig("result/total_firstmag_24_8_32_128_512_11dim_{:04d}.png".format(seed))
    # # plt.show()
    # plt.clf()