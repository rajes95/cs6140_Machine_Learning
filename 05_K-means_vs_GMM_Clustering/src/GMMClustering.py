'''
CS 6140 Machine Learning - Assignment 05

Problem 4 - GMM Clustering

@author: Rajesh Sakhamuru
@version: 8/5/2020
'''
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys


def distance(pnt1, pnt2):
    return math.sqrt(((pnt1 - pnt2) ** 2).sum())


def chooseCentroids(centroids, originalData):
    data = originalData.copy()
    centroidList = []
    for _, dataRow in data.iterrows():
        lowestDistance = math.inf
        bestCentroid = None

        for centroidIdx, centroid in centroids.iterrows():
            dist = distance(centroid, dataRow)
            if dist < lowestDistance:
                bestCentroid = centroidIdx
                lowestDistance = dist

        centroidList.append(bestCentroid)

    data["Centroid"] = centroidList
    return data


def updateCentroids(data, k):

    newCents = pd.DataFrame()

    for n in range(k):
        cent = data.loc[data["Centroid"] == n]
        cent = cent.drop(columns=["Centroid"])
        newCentroid = cent.mean(axis=0)
        newCents = newCents.append(newCentroid, ignore_index=True)

    return newCents


def clusterSSEs(data, newCentroids, k, clusterNotCentroid=False):
    SSEs = []

    for n in range(k):
        if clusterNotCentroid:
            centData = data.loc[data["Cluster"] == n]
            centData = centData.drop(columns=["Cluster"])
        else:
            centData = data.loc[data["Centroid"] == n]
            centData = centData.drop(columns=["Centroid"])

        diff = pd.DataFrame(centData.values - newCentroids.loc[[n]].values, columns=centData.columns)
        diffSqr = diff ** 2
        SSEs.append(diffSqr.sum(axis=1).sum())

    return SSEs


def entropy(data, target):
    '''
    Calculates entropy of a dataset in regards to a specific target column
   
    :param data: dataset as tuple for memoization
    :param target: target column name
    '''

    uniqueColCategories = data[target].unique()

    totalEntries = len(data)
    entropyS = 0
    for s in uniqueColCategories:
        subTable = data.loc[data[target] == s]
        tableLen = len(subTable)
        probability = (tableLen / totalEntries)
        entropyS -= probability * np.log2(probability)

    return (entropyS)


def NMI(dataWithClusterLabels, y_data, k):

    x_dataWithCentroids = dataWithClusterLabels
    x_dataWithCentroids["ActualLabel"] = y_data
#     print(x_dataWithCentroids)
    actualEntropy = entropy(x_dataWithCentroids, "ActualLabel")

    clusterEntropy = entropy(x_dataWithCentroids, "Cluster")

    conditionalEntropies = 0

    for n in range(k):
        # isolate data with the same centroid
        centData = x_dataWithCentroids.loc[x_dataWithCentroids["Cluster"] == n]
        conditionalEntropies += entropy(centData, "ActualLabel") * (len(centData) / len(x_dataWithCentroids))

    nmi = (2 * (actualEntropy - (conditionalEntropies))) / (actualEntropy + clusterEntropy)

    return nmi


def kMeansClustering(k, x_train):
    maxIterations = 100
    centroids = x_train.sample(n=k).reset_index(drop=True)

    for i in range(maxIterations):
        x_trainWithCentroids = chooseCentroids(centroids, x_train)
        newCentroids = updateCentroids(x_trainWithCentroids, k)

        if centroids.equals(newCentroids):
            break

        centroids = newCentroids

        if i == maxIterations - 1:
            print("max iterations reached")

    x_trainWithCentroids = chooseCentroids(newCentroids, x_train)

    SSEs = clusterSSEs(x_trainWithCentroids, newCentroids, k)

    return newCentroids, SSEs


def minMaxNormalize(data, x_trainMin, x_trainMax):
    return (data - x_trainMin) / (x_trainMax - x_trainMin)


def multivariate_normal(x, mean, covariance):
    d = len(x)
    x_m = x - mean

    prob = (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) *
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

    return prob[0][0]


def bestCluster(originalData, dataProbabilities):

    data = originalData.copy()

    clustList = []

    for p in dataProbabilities:
        p = p.tolist()
        clustList.append(p.index(max(p)))

    data["Cluster"] = clustList

    return data


def clusterProbabilities(originalData, clusterMeans, covariances, clusterDensity):
    """E-step"""
    data = originalData.copy()
    dataProbabilities = np.array([])
    for _, dataRow in data.iterrows():
        probList = []

        for meanIdx, mean in clusterMeans.iterrows():

            x = np.transpose(np.array([dataRow]))
            m = np.transpose(np.array([mean]))
            cov = np.array(covariances[meanIdx])
            
            probList.append(multivariate_normal(x, m, cov))

        probList = np.array(probList)
        clusterDensity = np.array(clusterDensity)

#         print(probList)
#         print(clusterDensity)

        clusterProbNumerators = (probList * clusterDensity)
        clusterProbs = (clusterProbNumerators / np.sum(clusterProbNumerators))

        if len(dataProbabilities) == 0:
            dataProbabilities = np.array([clusterProbs])
        else:
            dataProbabilities = np.append(dataProbabilities, [clusterProbs], axis=0)

    dataWithClusterLabels = bestCluster(data, dataProbabilities)

    return dataWithClusterLabels, dataProbabilities


def getNewMean(dataProbabilities, dataWithClusterLabels, k):

    data = dataWithClusterLabels.drop(columns=["Cluster"])

    newMeans = pd.DataFrame(columns=range(len(data.columns)))

    for n in range(k):
        cent = dataWithClusterLabels.loc[dataWithClusterLabels["Cluster"] == n]
        cent = cent.drop(columns=["Cluster"])
        weightedCentAvg = np.array([0] * len(data.columns))

        for idx, dataRow in cent.iterrows():
            weightedRow = dataRow * dataProbabilities[idx][n]
            weightedCentAvg = (weightedCentAvg + weightedRow)

        weightedCentAvg = weightedCentAvg / len(cent)
        newMeans = newMeans.append(weightedCentAvg, ignore_index=True)

    return newMeans


def getCovarianceMatrices(dataProbabilities, dataWithClusterLabels, newMean, k):

    covariances = []

    for n in range(k):
        clusterData = dataWithClusterLabels.loc[dataWithClusterLabels["Cluster"] == n]
        clusterData = clusterData.drop(columns=["Cluster"])

        if len(clusterData) <= 1:
            clusterData = clusterData.append(clusterData)

        weightedClusterStdDev = np.array([])

        for idx, dataRow in clusterData.iterrows():

            x_u = np.array([dataRow - newMean.loc[n]])

            oneRowVariance = dataProbabilities[idx][n] * np.dot(x_u.T, x_u)

            if len(weightedClusterStdDev) == 0:
                weightedClusterStdDev = oneRowVariance
            else:
                weightedClusterStdDev = weightedClusterStdDev + oneRowVariance

        weightedClusterStdDev = weightedClusterStdDev / len(clusterData)

        np.fill_diagonal(weightedClusterStdDev, weightedClusterStdDev.diagonal() + (10 ** -6))
        covariances.append(weightedClusterStdDev)

    return covariances


def gmmClustering(data, initialClusterMeans):
    clusterMeans = initialClusterMeans
    k = len(clusterMeans)

    dataWithEstimateCentroids = chooseCentroids(clusterMeans, data)
    clusterDensity = dataWithEstimateCentroids["Centroid"].value_counts(normalize=True).sort_index()

    covariances = []

    for n in range(k):
        centData = dataWithEstimateCentroids.loc[dataWithEstimateCentroids["Centroid"] == n]
        centData = centData.drop(columns=["Centroid"])

        if len(centData) <= 1:
            centData = centData.append(centData)
            
        covariance = np.array(centData.cov())
        np.fill_diagonal(covariance, covariance.diagonal() + (10 ** -6))
        covariances.append(covariance)

    for _ in range(15):
        dataWithClusterLabels, dataProbabilities = \
                clusterProbabilities(data, clusterMeans, covariances, clusterDensity)

        newDensity = dataWithClusterLabels["Cluster"].value_counts(normalize=True).sort_index()
        newMean = getNewMean(dataProbabilities, dataWithClusterLabels, k)
        newCovariances = getCovarianceMatrices(dataProbabilities, dataWithClusterLabels, newMean, k)

        clusterDensity = newDensity
        clusterMeans = newMean
        covariances = newCovariances

    SSEs = clusterSSEs(dataWithClusterLabels, clusterMeans, k, clusterNotCentroid=True)

    return clusterDensity, clusterMeans, covariances, dataWithClusterLabels, SSEs


def findBestK(x_train, y_train, numClasses):

    kValues = range(1, round(numClasses * 1.5))
    SSEList = []
    NMIList = []

    # cross validation of k values to pick the best one.
    # Also allows graphing of the k vs SSEs
    for k in kValues:

        avgSSE = 0
        avgNMI = 0
        for _ in range(2):
            # min-max feature normalization in order to prevent features with large
            # values from having unwanted bias
            # NEW DATA points can be scaled using the min and max values here
            x_trainMin = x_train.min()
            x_trainMax = x_train.max()
            normX_train = minMaxNormalize(x_train, x_trainMin, x_trainMax)

            normX_train = normX_train.fillna(0.5)

            kCentroids, SSEs = kMeansClustering(k, normX_train)

            clusterMeans = kCentroids.copy()
            _, clusterMeans, _, dataWithClusterLabels, SSEs = \
                            gmmClustering(normX_train, clusterMeans)

        #     pd.set_option('display.max_rows', None)
            nmi = NMI(dataWithClusterLabels, y_train, k)

            avgSSE += np.sum(SSEs)
            avgNMI += nmi

        avgSSE = avgSSE / 2
        avgNMI = avgNMI / 2
        print("k-value:", k)
        print("    Avg SSE:", avgSSE)
        print("    Avg NMI:", avgNMI)
        print()
        SSEList.append(avgSSE)
        NMIList.append(avgNMI)

    plot(kValues, SSEList, "Yeast Data: k-values vs SSE", "k-value", "SSE")
    plot(kValues, NMIList, "Yeast Data: k-values vs NMI", "k-value", "NMI")


def plot(xAxis, yVals, title, xLabel, yLabel):
    '''
    plots X vs Y graph given data, title, and axis labels
    
    :param xAxis: x data
    :param yVals: y data
    :param title: graph title
    :param xLabel: x axis label
    :param yLabel: y axis label
    '''

    plt.plot(xAxis, yVals, 'bo')
    plt.title(title)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()


def main():
    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    np.set_printoptions(linewidth=np.inf, suppress=True, threshold=sys.maxsize)

    colNames = range(int(sys.argv[2]))
    dataPath = str(sys.argv[1])
    numClasses = int(sys.argv[3])

    # import data file
    try:
        dataSet = pd.read_csv(dataPath, names=colNames)
    except:
        print("File not found")

    print("Data Set:")
    print(dataSet, "\n")

    target = len(colNames) - 1

    y_train = dataSet[target]
    x_train = dataSet.drop(target, axis=1)

    findBestK(x_train, y_train, numClasses)

"""
















"""
main()
