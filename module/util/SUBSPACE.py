import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import math

class SubspaceDiff():
    def __init__(self):
        pass
    
    def calc_karcher_subspace(self, basis1, basis2, dim):

        G = basis1@np.transpose(basis1) + basis2@np.transpose(basis2)
        a, l = np.linalg.eigh(G)
        alphas = a[::-1]
        lambdas = l[:, ::-1]
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
        return 2 * (len(s) - np.sum(s))
    
    def calc_rbf_magnitude(self, alphas1, alphas2, km):
        _, s, _ = np.linalg.svd(np.transpose(alphas1) @ km @ alphas2)
        return 2 * (len(s) - np.sum(s))