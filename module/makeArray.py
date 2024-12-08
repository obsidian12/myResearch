# ********************************************************************************************
# 파일명 : makeArray.py
# 목적　 : 파일로써 저장된 웨이트 파일을 다양한 형태로 불러오는 모듈
# 구조 　: 함수 구조(makeArray, makeDataArrayInSet, concatDataArray, dimrReduction)
# ********************************************************************************************

import numpy as np
import linecache
from glob import glob

# unitNumCounter : means number of total training step(in detail, means total number of sampled weight vector) 
# np.transpose(dataArray) : column vector equals weight vector of each training step,
#                           and each colmn number corresponds training step

# structure of csv file of weight data
# (...)→(direction which weightNum increases)
#   ↓
#   (direction which sampleNum increases)
# ex. sampling 200-sized weight in 360 times
# => csv file will be 360 x 200 matrix!

# makeDataArrayInSampleNum: 차원은 그대로, sampleNum(axis=1) 방향으로 행렬을 쌓은 뒤 전치
# ex. 360 x 200 사이즈(200-sized weight in 360 times sampling)의 csv file 10개에 대해 적용:
# 행렬은 200 x 3600 사이즈가 될 것이다!
# 층 별로 PCA 등을 행할 때 사용하면 됨
def makeDataArrayInSampleNum(csvFileList): # 여기서 FileList란 학습 한번에 저장된 웨이트 파일 리스트를 말하는 거임
    unitNumCounter = 0
    dataList = []

    for csvFile in csvFileList:
        f = open(csvFile, "r", encoding='CP932')
        while True:
            line = f.readline()
            if not line: break
            if csvFileList.index(csvFile) == 0: unitNumCounter = unitNumCounter + 1
            tmpList = line.split(",")
            returnList = []
            for element in tmpList:
                if element != "": returnList.append(float(element))
            dataList.append(returnList)
        f.close()

    dataArray = np.array(dataList)
    return unitNumCounter, np.transpose(dataArray)

# makeDataArrayInDim: sampleNum은 그대로, 차원(axis=0) 방향으로 행렬을 쌓은 뒤 전치
# ex. 360 x 200 사이즈(200-sized weight in 360 times sampling)의 csv file 10개에 대해 적용:
# 행렬은 2000 x 360 사이즈가 될 것이다!
# 각 층 별로 sampling 한 층별 weight를 전부 합칠 때 사용하면 됨
def makeDataArrayInDim(csvFileList): # 여기서 FileList란 학습 한번에 저장된 웨이트 파일 리스트를 말하는 거임
    unitNumCounter = 0 # in this mode, unitNumCounter equals to number of column of the matrix
    dataList = []

    for csvFile in csvFileList:
        f = open(csvFile, "r")
        tmpCounter = 0
        lineCounter = 0

        while True:
            line = f.readline()
            if line : lineCounter += 1
            if not line: break

            if lineCounter%10 != 0: continue

            if csvFileList.index(csvFile) == 0: 
                unitNumCounter = unitNumCounter + 1
                tmpList = line.split(",")
                returnList = []
                for element in tmpList:
                    if element != "": returnList.append(float(element))
                dataList.append(returnList)
            else:
                tmpList = line.split(",")
                for element in tmpList:
                    if element != "": dataList[tmpCounter].append(float(element))
            tmpCounter = tmpCounter + 1
        f.close()

    dataArray = np.array(dataList)
    return unitNumCounter, np.transpose(dataArray)

# makeDataArrayInDim: sampleNum은 그대로, 차원(axis=0) 방향으로 행렬을 쌓은 뒤 전치
# ex. 360 x 200 사이즈(200-sized weight in 360 times sampling)의 csv file 10개에 대해 적용:
# 행렬은 2000 x 360 사이즈가 될 것이다!
# 각 층 별로 sampling 한 층별 weight를 전부 합칠 때 사용하면 됨
def makeDataArrayInDimSpline(csvFileList, loadMin=0, loadMax=1048576): # 여기서 FileList란 학습 한번에 저장된 웨이트 파일 리스트를 말하는 거임
    unitNumCounter = 0 # in this mode, unitNumCounter equals to number of column of the matrix
    dataList = []

    for csvFile in csvFileList:
        for counter, lineNum in enumerate([loadMin, loadMax]):
            line = linecache.getline(csvFile, lineNum)
            if lineNum == 0 : print(line)
            if csvFileList.index(csvFile) == 0: 
                unitNumCounter = unitNumCounter + 1
                tmpList = line.split(",")
                returnList = []
                for element in tmpList:
                    if element != "": returnList.append(float(element))
                dataList.append(returnList)
            else:
                tmpList = line.split(",")
                for element in tmpList:
                    if element != "": dataList[counter].append(float(element))

    dataArray = np.array(dataList)
    return unitNumCounter, np.transpose(dataArray)

def makeDataArrayInDimSplines(csvFileList, loadlines): # 여기서 FileList란 학습 한번에 저장된 웨이트 파일 리스트를 말하는 거임
    unitNumCounter = 0 # in this mode, unitNumCounter equals to number of column of the matrix
    dataList = []

    for csvFile in csvFileList:
        for counter, lineNum in enumerate(loadlines):
            line = linecache.getline(csvFile, lineNum)
            if lineNum == 0 : print(line)
            if csvFileList.index(csvFile) == 0: 
                unitNumCounter = unitNumCounter + 1
                tmpList = line.split(",")
                returnList = []
                for element in tmpList:
                    if element != "": returnList.append(float(element))
                dataList.append(returnList)
            else:
                tmpList = line.split(",")
                for element in tmpList:
                    if element != "": dataList[counter].append(float(element))

    dataArray = np.array(dataList)
    return unitNumCounter, np.transpose(dataArray)

# makeDataArrayInSet: 주어진 stepNum 에 대해, csvFileList 내의 모든 웨이트들을 열 벡터로 쌓음
# ex. 360 x 200 사이즈(200-sized weight in 360 times sampling)의 csv file 10개에 대해 적용:
# 행렬은 10 x 360 사이즈가 될 것이다!
# 각 stepNum 별로 부분공간을 만들어 차분 부분공간을 구성할 때 사용하면 됨
def makeDataArrayInSet(csvFolderList, stepNum): # 여기서 FolderList란 학습 한번에 저장된 웨이트 파일들이 모여있는 폴더들의 리스트를 말하는 거임!!! 주의!!!
    dataList = []

    for csvFolderURL in csvFolderList:
        csvFileList = glob(csvFolderURL + "/layer*/part*.csv")
        returnList = []

        for csvFile in csvFileList: # 파일 리스트 안에 있는 
            f = open(csvFile, "r")
            for _ in range(stepNum + 1):
                line = f.readline()

            tmpList = line.split(",")
            
            for element in tmpList:
                if element != "": returnList.append(float(element))
            f.close()

        dataList.append(returnList)

    dataArray = np.array(dataList)
    return len(csvFolderList), np.transpose(dataArray)


# mode = 0: 차원은 그대로, sampleNum(axis=1) 방향으로 행렬을 쌓은 뒤 전치
# mode = 1: sampleNum은 그대로, 차원(axis=0) 방향으로 행렬을 쌓은 뒤 전치
def makeDataArray(csvFileList, mode=0):

    if mode == 0: return makeDataArrayInSampleNum(csvFileList)
    elif mode == 1: return makeDataArrayInDim(csvFileList)

# concatDataArrayInSampleNum: 차원은 그대로, sampleNum(axis=1) 방향으로 행렬을 쌓음
def concatDataArrayInSampleNum(dataArrayList):
    return np.concatenate(dataArrayList, axis=1)

# concatDataArrayInDim: sampleNum은 그대로, 차원(axis=0) 방향으로 행렬을 쌓음
def concatDataArrayInDim(dataArrayList):
    return np.concatenate(dataArrayList, axis=0)

# mode = 0: 차원은 그대로, sampleNum(axis=1) 방향으로 행렬을 쌓음
# mode = 1: sampleNum은 그대로, 차원(axis=0) 방향으로 행렬을 쌓음
def concatDataArray(dataArrayList, mode=0):

    if mode == 0: return concatDataArrayInSampleNum(dataArrayList)
    elif mode == 1: return concatDataArrayInDim(dataArrayList)

def dimReduction(csvFileList, newCsvFileList, ratio):
    for idx, csvFile in enumerate(csvFileList):
        #_, x = makeDataArray([csvFile])
        x = np.transpose(x)

        newX = x[::ratio,:]

        newRow, newColumn = newX.shape

        with open(newCsvFileList[idx], 'w') as f:
            for i in range(newRow):
                for j in range(newColumn):
                    if j == 0: f.write(str(newX[i,j]))
                    else: f.write("," + str(newX[i,j]))
                f.write("\n")