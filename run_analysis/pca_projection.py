import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from myResearch.module.util import PCA
from  myResearch.module import ParamIO
from glob import glob

def pca_projection(csvFolder, seed, n_components, gamma, kernelFlag=True):

    pcaTool = [PCA.WeightPCA(), PCA.WeightKPCA()]

    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    csvFileList = glob(csvFolder + "/layer*/part*.csv")
    _, dataArray = ParamIO.makeDataArrayInDim(csvFileList)

    if kernelFlag:
        projectedDataArray = pcaTool[1].rbf_kpca_proj(dataArray, n_components, gamma)
        projectedDataArray = np.transpose(projectedDataArray)
    else:
        projectedDataArray = pcaTool[0].pca_Proj(dataArray, n_components=n_components)
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
    if kernelFlag: plt.savefig("./result/kpca_weight_dim{:02d}_{:04d}.png".format(n_components, seed))
    else : plt.savefig("./result/linpca_weight_dim{:02d}_{:04d}.png".format(n_components, seed))
    plt.show()

def pca_projection_lowcost(csvFolder, seed, n_components, gamma, kernelFlag=True):

    pcaTool = [PCA.WeightPCA(), PCA.WeightKPCA()]

    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    csvFileList = glob(csvFolder + "/layer*/part*.csv")

    if kernelFlag:
        projectedDataArray = pcaTool[1].rbf_kpca_proj_lowcost(csvFileList, n_components, gamma)
        projectedDataArray = np.transpose(projectedDataArray)
    else:
        projectedDataArray = pcaTool[0].pca_Proj_lowcost(csvFileList, n_components=n_components)
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
    if kernelFlag: plt.savefig("./result/kpca_weight_dim{:02d}_{:04d}.png".format(n_components, seed))
    else : plt.savefig("./result/linpca_weight_dim{:02d}_{:04d}.png".format(n_components, seed))
    plt.show()