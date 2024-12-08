import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import module.pcaModule as wp

# analNum = 5

# f = open("result{}_0121.csv".format(analNum), "r")
# tmp1 = f.readline().split(",")
# tmp2 = f.readline().split(",")
# grp1 = [float(element) for element in tmp1]
# grp2 = [float(element) for element in tmp2]
# x = np.arange(len(grp1))

# plt.plot(x, grp1, label="without param initialization")
# plt.plot(x, grp2, label="with param initialization")
# plt.title("effect of parameter initializaion(d = 100)")
# plt.legend()
# plt.grid()
# plt.savefig("./result{}_0121_graph.png".format(analNum))
# plt.show()

fileName = sys.argv[1] + ".csv"
plotType = "C"

colorList = ["blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory",
             "blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory",
             "blue", "orange", "red", "purple", "green", 
             "olive", "brown", "grey", "cyan", "pink",
             "navy", "lime", "black", "yellow", "crimson",
             "gold", "skyblue", "indigo", "darkgreen", "ivory"]

def matrixToGraph(x, axis, fileName):

    if axis == 0: rowToGraph(x, fileName)
    elif axis == 1: rowToGraph(np.transpose(x), fileName)

def rowToGraph(x, fileName):

    graphNum = x.shape[0]
    space = np.arange(x.shape[1])

    for i in range(graphNum):
        plt.plot(space, x[i,:], color=colorList[i], label="graph {}".format(i))
    plt.legend()
    plt.grid()
    plt.savefig("graph_c" + fileName + ".png")
    plt.show()

def printMeanVar(fileName):
    f = open(fileName, "r")
    dataList = []
    while True:
        line = f.readline()
        if not line: break
        dataList.append(line.split(","))
    
    dimList = [70]
    initFlagList = ["True", "False"]
    stepList = [50, 100]
    for dim in dimList:
        for initFlag in initFlagList:
            for step in stepList:
                tmp = []
                for data in dataList:
                    if int(data[0]) > 3000 and int(data[0]) < 4000 and int(data[3]) == step and int(data[4]) == dim and data[6] == initFlag:
                        tmp.append(float(data[7]))
                #print(tmp)
                mean = sum(tmp) / len(tmp)
                var = 0
                for element in tmp:
                    var = var + (element - mean)*(element - mean)
                var = var / len(tmp)
                print("dim : {}, initFlag : {}, step : {} => mean : {}".format(dim, initFlag, step, mean))
                print("dim : {}, initFlag : {}, step : {} => square root var : {}".format(dim, initFlag, step, np.sqrt(var)))


# _, x = wp.makeDataArray([fileName])

# if plotType == "R": matrixToGraph(x, 0, fileName)
# elif plotType == "C": matrixToGraph(x, 1, fileName)
                
printMeanVar(fileName)