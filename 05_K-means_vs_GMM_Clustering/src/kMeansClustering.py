'''
CS 6140 Machine Learning - Assignment 05

Problem 3 - K-means Clustering

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


def clusterSSEs(data, newCentroids, k):
    SSEs = []

    for n in range(k):
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


def NMI(x_data, y_data, kCentroids, k):

    x_dataWithCentroids = chooseCentroids(kCentroids, x_data)
    x_dataWithCentroids["ActualLabel"] = y_data
#     print(x_dataWithCentroids)
    actualEntropy = entropy(x_dataWithCentroids, "ActualLabel")

    clusterEntropy = entropy(x_dataWithCentroids, "Centroid")

    conditionalEntropies = 0

    for n in range(k):
        # isolate data with the same centroid
        centData = x_dataWithCentroids.loc[x_dataWithCentroids["Centroid"] == n]
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


def findBestK(x_train, y_train, numClasses):

    kValues = range(1, round(numClasses * 1.5) + 1)
#     kValues = [3]
    SSEList = []
    NMIList = []
    
    # cross validation of k values to pick the best one. 
    # Also allows graphing of the k vs SSEs
    for k in kValues:

        avgSSE = 0
        avgNMI = 0
        for _ in range(5):
            # min-max feature normalization in order to prevent features with large
            # values from having unwanted bias
            # NEW DATA points can be scaled using the min and max values here
            x_trainMin = x_train.min()
            x_trainMax = x_train.max()
            normX_train = minMaxNormalize(x_train, x_trainMin, x_trainMax)

    #         print(normX_train)

            kCentroids, SSEs = kMeansClustering(k, normX_train)

        #     pd.set_option('display.max_rows', None)
            nmi = NMI(normX_train, y_train, kCentroids, k)

            avgSSE += np.sum(SSEs)
            avgNMI += nmi

        avgSSE = avgSSE / 5
        avgNMI = avgNMI / 5
        print("k-value:", k)
        print("    Avg SSE:", avgSSE)
        print("    Avg NMI:", avgNMI)
        print()
        SSEList.append(avgSSE)
        NMIList.append(avgNMI)

    plot(kValues, SSEList, "Soybean Data: k-values vs SSE", "k-value", "SSE")


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
