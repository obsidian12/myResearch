import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import module.ParamIO as ParamIO
import module.util.PCA as PCA
import module.util.SUBSPACE as SUBSPACE
from glob import glob
from tqdm import tqdm

def pca_projection_layerwise(pcaTool, csvFolder, seed, n_components, layerNum, sublayerNum, gamma, kernelFlag=True):

    if sublayerNum > 2: sln = sublayerNum + 1
    else : sln = sublayerNum

    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if layerNum == 0: csvFileList = glob(csvFolder + "/layer1/part*.csv")
    elif layerNum == 5: csvFileList = glob(csvFolder + "/layer22/part*.csv")
    else: csvFileList = glob(csvFolder + "/layer{}/part*.csv".format(layerNum*5 + sln - 4))
    
    _, dataArray = ParamIO.makeDataArray(csvFileList, mode=1)
    print(dataArray.shape)

    if kernelFlag:
        projectedDataArray = pcaTool.rbf_kpca_proj(dataArray, n_components, gamma)
        projectedDataArray = np.transpose(projectedDataArray)
    else:
        projectedDataArray = pcaTool.pca_Proj(dataArray, n_components=n_components)
        projectedDataArray = np.transpose(projectedDataArray)

    if n_components == 2:
        x = projectedDataArray[:,0]
        y = projectedDataArray[:,1]
        color = np.linspace(0, len(x), len(x))
        plt.scatter(x, y, c=color, edgecolor='none', s=3)
        plt.colorbar()
        plt.scatter(x[::int(len(x)/20)], y[::int(len(y)/20)], color=['gainsboro', 'mistyrose', 'pink', 'lightcoral', 'indianred', 'red', 'crimson', 'brown', 'darkred', 'black']*2, s=15)
    elif n_components == 3:
        x = projectedDataArray[:,0]
        y = projectedDataArray[:,1]
        z = projectedDataArray[:,2]
        color = np.linspace(0, len(x), len(x))
        ax.scatter(x, y, z, c=color, edgecolor='none', s=3)
        ax.scatter(x[::int(len(x)/20)], y[::int(len(y)/20)], z[::int(len(z)/20)], color=['gainsboro', 'mistyrose', 'pink', 'lightcoral', 'indianred', 'red', 'crimson', 'brown', 'darkred', 'black']*2, s=30, alpha=1)

    titleText = "weight transition visualization by PCA ( "
    titleText += "dim = {}".format(n_components)
    if kernelFlag : titleText += ", gamma = {}".format(gamma)
    titleText += " )"
    plt.title(titleText)
    if kernelFlag: plt.savefig("./result/kpca_layer{:02d}{:02d}_dim{:02d}_{:04d}.png".format(layerNum, sublayerNum, n_components, seed))
    else : plt.savefig("./result/linpca_layer{:02d}{:02d}_dim{:02d}_{:04d}.png".format(layerNum, sublayerNum, n_components, seed))
    plt.show()