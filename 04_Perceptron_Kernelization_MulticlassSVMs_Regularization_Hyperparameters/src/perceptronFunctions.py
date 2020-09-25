'''
CS 6140 Machine Learning - Assignment 04

Problem 1 - Perceptron and Dual Perceptron

@author: Rajesh Sakhamuru
@version: 7/20/2020
'''
import numpy as np
import pandas as pd
import statistics as st
import math


def getColNames(data):
    '''
    Returns list of column names of data given
    :param data: pandas dataframe
    '''
    return list(data.columns)


def zScoreNormalization(originalData, target, normParams=None):
    '''
    z-score normalization of features in data parameter. If mean and standard
    deviation are provided for each feature in normParams, then those are used to
    normalize the data.
    
    :param originalData: pandas dataframe
    :param target: target column name
    :param normParams: list of (mean, stdDev) tuples for each feature
    '''

    data = originalData.copy()

    colNames = getColNames(data)

    if target in colNames:
        colNames.remove(target)

    testing = False

    minMaxParams = []

    # if using given normalization parameters to normalize data
    if normParams is None:
        normParams = []
    else:
        testing = True

    count = 0

    for col in colNames:
        if col == '0':
            normParams.append((1, 0))
            count += 1
            continue
        # min and max feature values for feature re-scaling if wanted
        minMaxParams.append((data[col].min(), data[col].max()))

        if len(data[col].unique()) == 1:
            data[col] = 0

            if not testing:
                normParams.append((0, 0))

            count += 1
            continue
        if not testing:
            mean = data[col].mean()
            stdDev = st.pstdev(list(data[col]))
            normParams.append((mean, stdDev))
        else:
            mean, stdDev = normParams[count]

        data[col] = data[col] - mean
        data[col] = round(data[col] / stdDev, 3)

        count += 1

    return data, normParams, minMaxParams


def perceptron(originalData, target, learningRate):
    '''
    Calculates the weights to assign to features using the perceptron algorithm.
    
    :param originalData: pandas dataframe data
    :param target: target column name
    :param learningRate: learning rate to limit gradient descent jumps
    '''

    data = originalData.copy()

    colNames = getColNames(data)

    featList = colNames.copy()
    if target in colNames:
        featList.remove(target)

    weights = [[0]] * len(featList)

    weightsMatrix = np.array(weights)

    targetCol = data[target].values.tolist()
    targetMatrix = np.array([[n] for n in targetCol])

    dataMatrix = np.array(data.drop(columns=[target]))

    converged = False

    count = 0
    while not converged:
        # calculate predicted values using current weights
        predictedMatrix = np.dot(dataMatrix, weightsMatrix)
        updateVals = predictedMatrix * targetMatrix

        oldWeights = weightsMatrix
        weightsMatrix = np.transpose(weightsMatrix)

        for n in range(len(updateVals)):
            # if <=0 then the prediction was wrong
            if updateVals[n][0] <= 0:
                weightsMatrix = weightsMatrix + (learningRate * targetMatrix[n][0] * dataMatrix[n])

        weightsMatrix = np.transpose(weightsMatrix)

        count += 1

        comparison = (oldWeights == weightsMatrix)
        if comparison.all():
            converged = True

        # MAXIMUM of 2000 iterations
        if count >= 2000:
            converged = True

    return weightsMatrix


def dualPerceptron(originalData, target, learningRate):
    '''
    Calculates the weights to assign to features using dual perceptron algorithm. Also
    returns LogLoss list of each LogLoss value per iteration.
     
    :param originalData: pandas dataframe data
    :param target: target column name
    :param learningRate: learning rate to limit gradient descent jumps
    '''

    data = originalData.copy()

    colNames = getColNames(data)

    featList = colNames.copy()
    if target in colNames:
        featList.remove(target)

    weights = [[0]] * len(featList)

    alphas = np.array([[0]] * len(originalData))

    weightsMatrix = np.array(weights)

    targetCol = data[target].values.tolist()
    targetMatrix = np.array([[n] for n in targetCol])

    dataMatrix = np.array(data.drop(columns=[target]))

    converged = False

    count = 0
    while not converged:

        oldWeights = weightsMatrix

        # represent weights in Dual form
        unsummedWeights = learningRate * alphas * targetMatrix * dataMatrix
        summedWeights = np.sum(unsummedWeights, axis=0)
        weightsMatrix = np.transpose([summedWeights])

        # calculate predicted values using current weights
        predictedMatrix = np.dot(dataMatrix, weightsMatrix)
        updateVals = predictedMatrix * targetMatrix

        for n in range(len(updateVals)):
            # if <=0 then the prediction was wrong
            if updateVals[n][0] <= 0:
                alphas[n][0] = alphas[n][0] + 1

        count += 1

        comparison = (oldWeights == weightsMatrix)
        if comparison.all() and count >= 2:
            converged = True

        # MAXIMUM of 2000 iterations
        if count >= 2000:
            converged = True

    return weightsMatrix


def dualPerceptronKernel(originalData, target, kernel, gamma=None):
    '''
    Calculates the weights to assign to features using gradient descent. Also
    returns LogLoss list of each LogLoss value per iteration.
     
    :param originalData: pandas dataframe data
    :param target: target column name
    :param kernel: either linear or RBF
    :param gamma: for the RBF kernel
    '''
    data = originalData.copy()

    colNames = getColNames(data)

    featList = colNames.copy()
    if target in colNames:
        featList.remove(target)

    weights = [[0]] * len(featList)

    alphas = np.array([[0]] * len(originalData))

    weightsMatrix = np.array(weights)

    targetCol = data[target].values.tolist()
    targetMatrix = np.array([[n] for n in targetCol])

    dataMatrix = np.array(data.drop(columns=[target]))

    converged = False

    count = 0
    while not converged:

        oldWeights = weightsMatrix
        predictMatrix = []
        for i in range(len(dataMatrix)):
            rowIPred = 0

            for j in range(len(dataMatrix)):
                if gamma == None:
                    rowIPred += (alphas[j][0] * targetMatrix[j][0] * kernel(dataMatrix[j], dataMatrix[i]))
                else:
                    rowIPred += (alphas[j][0] * targetMatrix[j][0] * kernel(dataMatrix[j], dataMatrix[i], gamma=gamma))

            predictMatrix.append([rowIPred])

        updateVals = predictMatrix * targetMatrix
        for n in range(len(updateVals)):
            # if <=0 then the prediction was wrong
            if updateVals[n][0] <= 0:
                alphas[n][0] = alphas[n][0] + 1

        count += 1

        comparison = (oldWeights == weightsMatrix)
        if comparison.all() and count >= 2:
            converged = True

        # MAXIMUM of 500 iterations
        if count >= 500:
            converged = True

    return alphas


def linearKernel(trainMatrix, testMatrix):
    '''
    Linear kernel function passed to perceptron algo
    :param trainMatrix: train matrix being kernelized
    :param testMatrix: test matrix being kernelized
    '''
    return np.dot(trainMatrix, testMatrix)


def gaussianKernel(trainMatrix, testMatrix, gamma=4):
    '''
    Gaussian RBF Kernel for perceptron algo
    :param trainMatrix: matrix being kernelized
    :param testMatrix: matrix being kernelized
    :param gamma: kernel gamma value
    '''

    k = math.exp(-(gamma) * (np.dot((trainMatrix - testMatrix), (trainMatrix - testMatrix))))

    return k


def alphasPredict(testMatrix, alphas, trainMatrix, trainTargetMatrix, kernel, gamma=None):
    '''
    Predict using alphas instead of weights for dual form algorithm
    :param testMatrix: Testing Data Matrix
    :param alphas: Alphas related to Training Data
    :param trainMatrix: Training Data matrix
    :param trainTargetMatrix: Training Target Matrix
    :param kernel: Which kernel is used in the perceptron algo
    :param gamma: gamma used in perceptron algo
    '''
    predictMatrix = []
    for i in range(len(testMatrix)):
        rowIPred = 0

        for j in range(len(trainMatrix)):
            if gamma == None:
                rowIPred += (alphas[j][0] * trainTargetMatrix[j][0] * kernel(trainMatrix[j], testMatrix[i]))
            else:
                rowIPred += (alphas[j][0] * trainTargetMatrix[j][0] * kernel(trainMatrix[j], testMatrix[i], gamma=gamma))

        predictMatrix.append([rowIPred])

    return predictMatrix


def calcAccuracyUsingAlphas(testData, trainData, target, alphas, kernel, gamma=None):
    '''
    Accuracy using Alpha values is calculated
    :param testData: Data being predicted upon
    :param trainData: training data used for predictions
    :param target: target column
    :param alphas: alphas related to training data
    :param kernel: kernel used to generate alpha values during training
    :param gamma: gamma used when training to generate alpha values
    '''
    # split test features and test target column
    targetCol = testData[target].values.tolist()
    targetMatrix = np.array([[n] for n in targetCol])
    testMatrix = np.array(testData.drop(columns=[target]))

    # split the training data into feature columns and target column
    targetCol = trainData[target].values.tolist()
    trainTargetMatrix = np.array([[n] for n in targetCol])
    trainMatrix = np.array(trainData.drop(columns=[target]))

    predictedMatrix = alphasPredict(testMatrix, alphas, trainMatrix, trainTargetMatrix, kernel, gamma=gamma)

    # Empty confusion matrix
    targetCats = testData[target].unique()
    confusionDict = {}
    for cat in targetCats:
        confusionDict[cat] = [0] * len(targetCats)
    confusionMatrix = pd.DataFrame(data=confusionDict, index=targetCats)

    count = 0

    for n in range(len(predictedMatrix)):
        if predictedMatrix[n][0] < 0:
            prediction = -1
        else:
            prediction = 1
        actual = targetMatrix[n][0]

#         print("prediction:", prediction, "actual:", actual, "  ")

        confusionMatrix.at[(prediction), (actual)] = confusionMatrix.at[(prediction), (actual)] + 1

        if int(prediction) == int(actual):
            count += 1

    acc = count / len(predictedMatrix)

    return acc, confusionMatrix


def calculateAccuracy(data, target, weightsMatrix):
    '''
    Calculate how well the given weights matrix can predict the data.
    
    :param data: pandas dataframe with features and target column
    :param target: target column name
    :param weightsMatrix: weights matrix generated by gradient descent
    '''

    targetCol = data[target].values.tolist()
    targetMatrix = np.array([[n] for n in targetCol])

    dataMatrix = np.array(data.drop(columns=[target]))

    predictedMatrix = np.dot(dataMatrix, weightsMatrix)

    targetCats = data[target].unique()

    confusionDict = {}
    for cat in targetCats:
        confusionDict[cat] = [0] * len(targetCats)

    confusionMatrix = pd.DataFrame(data=confusionDict, index=targetCats)

    count = 0

    for n in range(len(predictedMatrix)):
        if predictedMatrix[n][0] < 0:
            prediction = -1
        else:
            prediction = 1
        actual = targetMatrix[n][0]

#         print("prediction:", prediction, "actual:", actual, "  ")

        confusionMatrix.at[(prediction), (actual)] = confusionMatrix.at[(prediction), (actual)] + 1

        if int(prediction) == int(actual):
            count += 1

    acc = count / len(predictedMatrix)

    return acc, confusionMatrix

