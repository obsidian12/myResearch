# ********************************************************************************************
# 파일명 : vecModule.py
# 목적　 : 벡터 1개(주로 웨이트 1개)에 대해 원소 간 평균, 분산, k 함수값, 차분 등을 계산 가능한 모듈
# 구조 　: 함수 구조(k 함수값, 평균, 분산, 차분)
# ********************************************************************************************
import numpy as np

def k_function(vec, r):
    d = vec.shape[0]

    norm = np.linalg.norm(vec, axis=0)
    normVec = vec / norm
    #print(normVec)

    sum = 0
    for i in range(d):
        for j in range(d):
            if i != j:
                dist = abs(normVec[i] - normVec[j])
                if dist <= r: sum = sum + 1

    return sum / (d*d)

def vec_mean(vec):

    d = vec.shape[0]
    mean = 0

    for i in range(d):
        mean = mean + vec[i] / d

    return mean

def vec_variance(vec):

    d = vec.shape[0]
    var = 0
    mean = vec_mean(vec)

    for i in range(d):
        var = var + (vec[i] - mean)*(vec[i] - mean)

    return var

def vec_numerical_diff(vec):
    n = len(vec)
    result = []
    for i in range(n):
        if i == 0: result.append(0)
        else: result.append(vec[i] - vec[i - 1])
    return result