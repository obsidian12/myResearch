# ********************************************************************************************
# 파일명 : ParamIO.py
# 목적　 : 파일로써 저장된 웨이트 파일을 다양한 형태로 불러오는 모듈
# 구조 　: 함수 구조(makeArray, makeDataArrayInSet, concatDataArray, dimrReduction)
# ********************************************************************************************

import numpy as np
import linecache

########################################################
#################### Important! ########################
######## structure of csv file of weight data ##########
# (...)→(direction which weightNum increases)
#   ↓
#   (direction which sampleNum increases)
# ex. sampling 200-sized weight in 360 times
# => csv file will be 360 x 200 matrix!
########################################################
########################################################

## makeDataArrayInSampleNum : sampleNum(axis=0) 방향으로 행렬을 쌓은 후 전치(즉, 다른 학습에서 sampling 되어진 웨이트 이력들을 통합)
######## makeDataArrayInDim : weightNum(axis=1) 방향으로 행렬을 쌓은 후 전치(즉, 매 sampleNum 마다의 모든 웨이트를 행벡터화)
##### makeDataArrayInDimSplines : 특정 라인만 가져올 수 있는 makeDataArrayInDim

######## sampleNumReduction : sampling 빈도가 과한 데이터 행렬들을 ratio 만큼 sampling 빈도를 줄여서 다시 저장

# makeDataArrayInSampleNum: 차원은 그대로, sampleNum(axis=0) 방향으로 행렬을 쌓은 뒤 전치
# ex. 360 x 200 사이즈(200-sized weight in 360 times sampling)의 csv file 10개에 대해 적용:
# 행렬은 200 x 3600 사이즈가 될 것이다!
# 층 별로 PCA 등을 행할 때 사용하면 됨
def makeDataArrayInSampleNum(csvFileList): # 여기서 FileList란 학습 한번에 저장된 웨이트 파일 리스트를 말하는 거임
    unitNumCounter = 0
    dataList = []

    if str(type(csvFileList)) == "<class 'str'>" : csvFileList = [csvFileList]

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

    if str(type(csvFileList)) == "<class 'str'>" : csvFileList = [csvFileList]

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

def makeDataArrayInDimSplines(csvFileList, loadlines): # 여기서 FileList란 학습 한번에 저장된 웨이트 파일 리스트를 말하는 거임
    unitNumCounter = 0 # in this mode, unitNumCounter equals to number of column of the matrix
    dataList = []
    if str(type(csvFileList)) == "<class 'str'>" : csvFileList = [csvFileList]
    if str(type(loadlines)) == "<class 'int'>": loadlines = [loadlines]

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

# sampleNumReduction : sampling 빈도가 과도한 csvFile에 대해,
# sampling 빈도를 ratio 만큼 줄여서 새로운 newCsvFile 에 저장하는 메소드
def sampleNumReduction(csvFileList, newCsvFileList, ratio):
    for idx, csvFile in enumerate(csvFileList):
        _, x = makeDataArrayInSampleNum([csvFile]) # csv 파일은 1개이므로, SampleNum 메소드를 써도 Dim 메소드를 써도 괜찮음
        x = np.transpose(x)

        newX = x[::ratio,:]

        newRow, newColumn = newX.shape

        with open(newCsvFileList[idx], 'w') as f:
            for i in range(newRow):
                for j in range(newColumn):
                    if j == 0: f.write(str(newX[i,j]))
                    else: f.write("," + str(newX[i,j]))
                f.write("\n")