# ********************************************************************************************
# 파일명 : smModule.py
# 목적　 : 부분공간 사이의 1차, 2차 차분 부분공간, magnitude 등을 계산 가능한 모듈
# 구조 　: 클래스 구조(분할 파일 DLDR 계산, 일반 파일 DLDR 계산, PCA 계산, 푸리에 변환)
# ********************************************************************************************
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import copy as cp
from glob import glob
from tqdm import tqdm
#import makeArray as ma
from scipy.spatial.distance import pdist, squareform


class SubspaceDiff():
    def __init__(self):
        pass
    
    def calc_karcher_subspace(self, basis1, basis2, dim):

        G = basis1@np.transpose(basis1) + basis2@np.transpose(basis2)
        a, l = np.linalg.eigh(G)
        alphas = a[::-1]
        lambdas = l[:, ::-1]
        # print(alphas)
        # print(alphas[0:dim])
        # print(alphas[dim:2*dim])
        return [alphas[0:dim], lambdas[:,0:dim]]
    
    def calc_diff_subspace(self, basis1, basis2, dim):

        G = basis1@np.transpose(basis1) + basis2@np.transpose(basis2)
        a, l = np.linalg.eigh(G)
        alphas = a[::-1]
        lambdas = l[:, ::-1]
        return [alphas[dim:2*dim], lambdas[:,dim:2*dim]]
    
    def calc_magnitude(self, basis1, basis2, tmp=False):
        G = np.transpose(basis1)@basis2
        _, s, _ = np.linalg.svd(G)
        overlappedNum = 0
        for i, element in enumerate(s) :
            if math.isclose(1.0, element) : continue
            else :
                overlappedNum = i + 1
                break
        if tmp :
            print("")
            if overlappedNum == 1: print("There is no dimensions overlapped!")
            else: print("{} dimensions are overlapped!".format(overlappedNum - 1))
            
        s = s[overlappedNum - 1:]
        if tmp : 
            print("s : {} ~ {}".format(s[0], s[-1]))
        # return 2 * (1 - s[0])
        # return 2 * (3 - np.sum(s[0:3]))
        return 2 * (len(s) - np.sum(s))
    
    def calc_rbf_magnitude(self, alphas1, alphas2, km):
        _, s, _ = np.linalg.svd(np.transpose(alphas1) @ km @ alphas2)
        return 2 * (len(s) - np.sum(s))
    
    def rbf_kernel(self, x_1, x_2, gamma):
        dist = np.linalg.norm(x_1 - x_2)
        return np.exp(-gamma * dist * dist)