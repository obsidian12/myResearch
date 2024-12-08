# ********************************************************************************************
# 파일명 : weight_norm_difference.py
# 목적　 : 1개의 weight trajectory에 대해 ***step 별로, layer 별로 weight의 노름의 변화, (정규화된) weight의 변화의 크기***을 계산,
#          step 별로 각 층의 weight가 어떤 궤적을 그리며 변화하는지 분석하는 모듈
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

def print_ip_difference_norm(seed, csvFolder, interval, totalSampleNum=1000):

    magContainer = []
    print("layer input")

    for i in tqdm(range(int(totalSampleNum / interval))):
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer1/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
        before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
        after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
        magContainer.append(np.linalg.norm(before - after))

    # plt.title("first-magnitude of weight subspace(dim=11, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_11dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def print_ip_norm_difference(seed, csvFolder, interval, totalSampleNum=1000):

    magContainer = []
    print("layer input")

    for i in tqdm(range(int(totalSampleNum / interval))):
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer1/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
        magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))

    # plt.title("first-magnitude of weight subspace(dim=11, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_11dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def print_bbi_difference_norm(seed, csvFolder, layerNum, interval, totalSampleNum=1000):

    magContainer = []
    print("layer {}(bb input)".format(layerNum))

    for i in tqdm(range(int(totalSampleNum / interval))):
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 1)), loadMin=interval*i + 1, loadMax=interval*(i+1))
        before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
        after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
        magContainer.append(np.linalg.norm(before - after))

    # plt.title("first-magnitude of weight subspace(dim=11, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_11dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def print_bbi_norm_difference(seed, csvFolder, layerNum, interval, totalSampleNum=1000):

    magContainer = []
    print("layer {}(bb input)".format(layerNum))

    for i in tqdm(range(int(totalSampleNum / interval))):
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), loadMin=interval*i + 1, loadMax=interval*(i+1))
        magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))

    # plt.title("first-magnitude of weight subspace(dim=11, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_11dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def print_bb_difference_norm(seed, csvFolder, layerNum, sublayerNum, interval, totalSampleNum=1000):

    magContainer = []
    print("layer {}(bb{}-th)".format(layerNum, sublayerNum))

    for i in tqdm(range(int(totalSampleNum / interval))):

        if sublayerNum == 1:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), loadMin=interval*i + 1, loadMax=interval*(i+1))
            before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
            after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
            magContainer.append(np.linalg.norm(before - after))
        elif sublayerNum == 2:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)), loadMin=interval*i + 1, loadMax=interval*(i+1))
            before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
            after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
            magContainer.append(np.linalg.norm(before - after))
        elif sublayerNum == 3:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), loadMin=interval*i + 1, loadMax=interval*(i+1))
            before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
            after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
            magContainer.append(np.linalg.norm(before - after))

    # plt.title("first-magnitude of weight subspace(dim=11, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_11dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def print_bb_norm_difference(seed, csvFolder, layerNum, sublayerNum, interval, totalSampleNum=1000):

    magContainer = []
    print("layer {}(bb{}-th)".format(layerNum, sublayerNum))

    for i in tqdm(range(int(totalSampleNum / interval))):

        if sublayerNum == 1:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 - 2)), loadMin=interval*i + 1, loadMax=interval*(i+1))
            magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))
        elif sublayerNum == 2:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5)), loadMin=interval*i + 1, loadMax=interval*(i+1))
            magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))
        elif sublayerNum == 3:
            _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + 1)), loadMin=interval*i + 1, loadMax=interval*(i+1))
            magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))

    # plt.title("first-magnitude of weight subspace(dim=11, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_11dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def print_fc_difference_norm(seed, csvFolder, interval, totalSampleNum=1000):

    magContainer = []
    print("layer fc")

    for i in tqdm(range(int(totalSampleNum / interval))):
        # _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer22/part*.csv") + glob(csvFolder + "/layer23/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer22/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
        before = weight_layer[:,0] / np.linalg.norm(weight_layer[:,0])
        after = weight_layer[:,1] / np.linalg.norm(weight_layer[:,1])
        magContainer.append(np.linalg.norm(before - after))

    # plt.title("first-magnitude of weight subspace(dim=11, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_11dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

def print_fc_norm_difference(seed, csvFolder, interval, totalSampleNum=1000):

    magContainer = []
    print("layer fc")

    for i in tqdm(range(int(totalSampleNum / interval))):
        # _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer22/part*.csv") + glob(csvFolder + "/layer23/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
        _, weight_layer = ma.makeDataArrayInDimSpline(glob(csvFolder + "/layer22/part*.csv"), loadMin=interval*i + 1, loadMax=interval*(i+1))
        magContainer.append(abs(np.linalg.norm(weight_layer[:,0]) - np.linalg.norm(weight_layer[:,1])))

    # plt.title("first-magnitude of weight subspace(dim=11, layer=fc)")
    # plt.xlabel("each step")
    # plt.ylabel("1st magnitude")
    # plt.plot(range(len(magContainer)), magContainer)
    # plt.savefig("result/layerfc_firstmag_11dim_{:04d}.png".format(seed))
    # plt.show()
    # plt.clf()

    return np.array(magContainer)

# ------------------------------------------------------------------

dataset = "CIFAR"
op = "norm"

if dataset == "MNIST":
    pass

elif dataset == "CIFAR":

    seed = 8311
    interval = 10

    csvFolderList = glob("C:/Users/dmtsa/research/run_analysis/DB_CNN_MNIST_1/1-4-8-fc__*")
    csvFolder = "C:/Users/dmtsa/research/run_analysis/DB_RESNET18_CIFAR_4/resnet18__{:04d}".format(seed)

    if op == "normdiff":
        layer = 0
        if layer == 0:
            i1 = print_ip_norm_difference(seed, csvFolder, interval)
            f1 = print_fc_norm_difference(seed, csvFolder, interval)

            plt.title("norm difference of weight(layer : input, fc)")
            plt.xlabel("each step")
            plt.ylabel("norm")
            plt.plot(range(len(i1)), i1, label="input layer")
            plt.plot(range(len(f1)), f1, label="fc layer")
            plt.grid()
            plt.legend()
            plt.savefig("result/ipfc_normdiff_interval{}_{:04d}(without_bias).png".format(interval, seed))
            # plt.show()
            plt.clf()
        elif layer == 1:
            n1 = print_bbi_norm_difference(seed, csvFolder, 1, interval)
            n2 = print_bbi_norm_difference(seed, csvFolder, 2, interval)
            n3 = print_bbi_norm_difference(seed, csvFolder, 3, interval)
            n4 = print_bbi_norm_difference(seed, csvFolder, 4, interval)

            plt.title("norm difference of weight(layer : sub-input layer(basicblock1~4))")
            plt.xlabel("each step")
            plt.ylabel("norm")
            plt.plot(range(len(n1)), n1, label="block 1(bb input)")
            plt.plot(range(len(n2)), n2, label="block 2(bb input)")
            plt.plot(range(len(n3)), n3, label="block 3(bb input)")
            plt.plot(range(len(n4)), n4, label="block 4(bb input)")
            plt.grid()
            plt.legend()
            plt.savefig("result/bbi_normdiff_interval{}_{:04d}.png".format(interval, seed))
            # plt.show()
            plt.clf()
        elif layer == 2:
            c11 = print_bb_norm_difference(seed, csvFolder, 1, 1, interval)
            c12 = print_bb_norm_difference(seed, csvFolder, 2, 1, interval)
            c13 = print_bb_norm_difference(seed, csvFolder, 3, 1, interval)
            c14 = print_bb_norm_difference(seed, csvFolder, 4, 1, interval)

            plt.title("norm difference of weight(layer : sub-first layer(basicblock1~4))")
            plt.xlabel("each step")
            plt.ylabel("norm")
            plt.plot(range(len(c11)), c11, label="block 1(bb first)")
            plt.plot(range(len(c12)), c12, label="block 2(bb first)")
            plt.plot(range(len(c13)), c13, label="block 3(bb first)")
            plt.plot(range(len(c14)), c14, label="block 4(bb first)")
            plt.grid()
            plt.legend()
            plt.savefig("result/bb1_normdiff_interval{}_{:04d}.png".format(interval, seed))
            # plt.show()
            plt.clf()
        elif layer == 3:
            c21 = print_bb_norm_difference(seed, csvFolder, 1, 2, interval)
            c22 = print_bb_norm_difference(seed, csvFolder, 2, 2, interval)
            c23 = print_bb_norm_difference(seed, csvFolder, 3, 2, interval)
            c24 = print_bb_norm_difference(seed, csvFolder, 4, 2, interval)

            plt.title("norm difference of weight(layer : sub-second layer(basicblock1~4))")
            plt.xlabel("each step")
            plt.ylabel("norm")
            plt.plot(range(len(c21)), c21, label="block 1(bb second)")
            plt.plot(range(len(c22)), c22, label="block 2(bb second)")
            plt.plot(range(len(c23)), c23, label="block 3(bb second)")
            plt.plot(range(len(c24)), c24, label="block 4(bb second)")
            plt.grid()
            plt.legend()
            plt.savefig("result/bb2_normdiff_interval{}_{:04d}.png".format(interval, seed))
            # plt.show()
            plt.clf()
        elif layer == 4:
            c31 = print_bb_norm_difference(seed, csvFolder, 1, 3, interval)
            c32 = print_bb_norm_difference(seed, csvFolder, 2, 3, interval)
            c33 = print_bb_norm_difference(seed, csvFolder, 3, 3, interval)
            c34 = print_bb_norm_difference(seed, csvFolder, 4, 3, interval)

            plt.title("norm difference of weight(layer : sub-third layer(basicblock1~4))")
            plt.xlabel("each step")
            plt.ylabel("norm")
            plt.plot(range(len(c31)), c31, label="block 1(bb third)")
            plt.plot(range(len(c32)), c32, label="block 2(bb third)")
            plt.plot(range(len(c33)), c33, label="block 3(bb third)")
            plt.plot(range(len(c34)), c34, label="block 4(bb third)")
            plt.grid()
            plt.legend()
            plt.savefig("result/bb3_normdiff_interval{}_{:04d}.png".format(interval, seed))
            # plt.show()
            plt.clf()
    
    elif op == "norm":
        layer = 0
        if layer == 0:
            i1 = print_ip_difference_norm(seed, csvFolder, interval)
            f1 = print_fc_difference_norm(seed, csvFolder, interval)

            plt.title("difference of normalized weight(layer : input, fc)")
            plt.xlabel("each step")
            plt.ylabel("norm")
            plt.plot(range(len(i1)), i1, label="input layer")
            plt.plot(range(len(f1)), f1, label="fc layer")
            plt.grid()
            plt.legend()
            plt.savefig("result/ipfc_norm_interval{}_{:04d}(without_bias).png".format(interval, seed))
            # plt.show()
            plt.clf()
        elif layer == 1:
            n1 = print_bbi_difference_norm(seed, csvFolder, 1, interval)
            n2 = print_bbi_difference_norm(seed, csvFolder, 2, interval)
            n3 = print_bbi_difference_norm(seed, csvFolder, 3, interval)
            n4 = print_bbi_difference_norm(seed, csvFolder, 4, interval)

            plt.title("difference of normalized weight(layer : sub-input layer(basicblock1~4))")
            plt.xlabel("each step")
            plt.ylabel("norm")
            plt.plot(range(len(n1)), n1, label="block 1(bb input)")
            plt.plot(range(len(n2)), n2, label="block 2(bb input)")
            plt.plot(range(len(n3)), n3, label="block 3(bb input)")
            plt.plot(range(len(n4)), n4, label="block 4(bb input)")
            plt.grid()
            plt.legend()
            plt.savefig("result/bbi_norm_interval{}_{:04d}.png".format(interval, seed))
            # plt.show()
            plt.clf()
        elif layer == 2:
            c11 = print_bb_difference_norm(seed, csvFolder, 1, 1, interval)
            c12 = print_bb_difference_norm(seed, csvFolder, 2, 1, interval)
            c13 = print_bb_difference_norm(seed, csvFolder, 3, 1, interval)
            c14 = print_bb_difference_norm(seed, csvFolder, 4, 1, interval)

            plt.title("difference of normalized weight(layer : sub-first layer(basicblock1~4))")
            plt.xlabel("each step")
            plt.ylabel("norm")
            plt.plot(range(len(c11)), c11, label="block 1(bb first)")
            plt.plot(range(len(c12)), c12, label="block 2(bb first)")
            plt.plot(range(len(c13)), c13, label="block 3(bb first)")
            plt.plot(range(len(c14)), c14, label="block 4(bb first)")
            plt.grid()
            plt.legend()
            plt.savefig("result/bb1_norm_interval{}_{:04d}.png".format(interval, seed))
            # plt.show()
            plt.clf()
        elif layer == 3:
            c21 = print_bb_difference_norm(seed, csvFolder, 1, 2, interval)
            c22 = print_bb_difference_norm(seed, csvFolder, 2, 2, interval)
            c23 = print_bb_difference_norm(seed, csvFolder, 3, 2, interval)
            c24 = print_bb_difference_norm(seed, csvFolder, 4, 2, interval)

            plt.title("difference of normalized weight(layer : sub-second layer(basicblock1~4))")
            plt.xlabel("each step")
            plt.ylabel("norm")
            plt.plot(range(len(c21)), c21, label="block 1(bb second)")
            plt.plot(range(len(c22)), c22, label="block 2(bb second)")
            plt.plot(range(len(c23)), c23, label="block 3(bb second)")
            plt.plot(range(len(c24)), c24, label="block 4(bb second)")
            plt.grid()
            plt.legend()
            plt.savefig("result/bb2_norm_interval{}_{:04d}.png".format(interval, seed))
            # plt.show()
            plt.clf()
        elif layer == 4:
            c31 = print_bb_difference_norm(seed, csvFolder, 1, 3, interval)
            c32 = print_bb_difference_norm(seed, csvFolder, 2, 3, interval)
            c33 = print_bb_difference_norm(seed, csvFolder, 3, 3, interval)
            c34 = print_bb_difference_norm(seed, csvFolder, 4, 3, interval)

            plt.title("difference of normalized weight(layer : sub-third layer(basicblock1~4))")
            plt.xlabel("each step")
            plt.ylabel("norm")
            plt.plot(range(len(c31)), c31, label="block 1(bb third)")
            plt.plot(range(len(c32)), c32, label="block 2(bb third)")
            plt.plot(range(len(c33)), c33, label="block 3(bb third)")
            plt.plot(range(len(c34)), c34, label="block 4(bb third)")
            plt.grid()
            plt.legend()
            plt.savefig("result/bb3_norm_interval{}_{:04d}.png".format(interval, seed))
            # plt.show()
            plt.clf()