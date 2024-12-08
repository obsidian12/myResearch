# ********************************************************************************************
# 파일명 : pca_exercise.py
# 목적　 : PCA 구현 과정 연습
# 구조 　: 일직선 구조(데이터셋 랜덤 생성, 자기상관행렬 계산, 고유값 분해, 고유공간 정사영, 플롯)
# ********************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. construct dataset
x_list = [[1, 5, 9, 2, 3], [5, 4, 2, 1, 3], [11, 15, 7, 4, 7], [8, 5, 6, 1, 11]]
x = np.array(x_list)
x = np.transpose(x)
x = x / 4

# 2. caculate auto-correlation matrix, and do eigen decomposition 
r = x@np.transpose(x)
l, U = np.linalg.eig(r)

# 3. get basis vector of eigenspace
Ulen = np.linalg.norm(U, axis=0)
for i in range(5):
    U[:,i] = U[:,i] / Ulen[i]

# 4. project data to eigenspace
result = np.full(3, 0)
for pcNum in range(3):
    result[pcNum] = np.dot(x[:,0], U[:,pcNum])
print(result)

for dataNum in range(1, 4):
    tmp = np.full(3, 0)
    for pcNum in range(3):
        tmp[pcNum] = np.dot(x[:,dataNum], U[:,pcNum])
    result = np.vstack([result, tmp])

result = np.transpose(result)
# 5. plot result
fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(111, projection='3d')
ax.plot(result[0,:], result[1,:], result[2,:], 'b')
plt.show()