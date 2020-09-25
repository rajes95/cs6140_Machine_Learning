'''
CS 6140 Machine Learning - Assignment 04

Problem 2 - Regularized Logistic Regression

@author: Rajesh Sakhamuru
@version: 7/20/2020
'''

import numpy as np
import pandas as pd
import statistics as st
import math


def tenFoldCrossValidation(originalData, target, learningRate, tolerance, verbose=True):
    '''
    This function generates weights using logistic regression through gradient descent
    and tests the accuracy of those weights based on 10 unique folds of the 
    dataset provided. Then the accuracy, precision and recall values over the 
    folds and the confusion matrix are returned.
    
    :param originalData: dataframe with all training and testing data 
    :param target: target column name
    :param learningRate: learning rate for gradient descent
    :param tolerance: tolerence threshold to determine when to stop gradient descent
    :param verbose: if True, will print testing results per fold
    '''
    # prevent changes to original data
    data = originalData.copy()

    # shuffle data
    dataShuffled = data.sample(frac=1).reset_index(drop=True)
    numRows = len(dataShuffled)

    oneTenthRows = round(numRows / 10)

    # list of loss values and accuracy, precison and recall data over the folds
    trainLogLossValues = []
    testLogLossValues = []
    trainAccuracyList = []
    testAccuracyList = []
    trainPrecisionList = []
    testPrecisionList = []
    trainRecallList = []
    testRecallList = []

    # picks best log loss training descent to be graphed based on best fold
    lowestLogLoss = math.inf
    bestFold = 0
    bestLogLossList = []

    # empty confusion matrix
    targetCats = data[target].unique()
    confusionDict = {}
    for cat in targetCats:
        confusionDict[cat] = [0] * len(targetCats)
    testConfusionMatrix = pd.DataFrame(data=confusionDict, index=targetCats)
    trainConfusionMatrix = pd.DataFrame(data=confusionDict, index=targetCats)

    # 10-fold cross validation of growTree model
    for n in range(10):

        # split into training and testing data
        testData = dataShuffled[(n * oneTenthRows):oneTenthRows * (n + 1)]
        trainData = dataShuffled.drop(dataShuffled.index[(n * oneTenthRows):oneTenthRows * (n + 1)])

        # normalize training and test data with zscore normalization
        normTrainData, normParams, _ = zScoreNormalization(trainData, target)
        normTestData, _, _ = zScoreNormalization(testData, target, normParams=normParams)

        # calculate weights using gradient descent (and return gradient descent log loss list)
        weightsMatrix, logLossList = gradientDescent(normTrainData, target, learningRate, tolerance)

        # calculate training and testing accuracy and confusion matrices
        trainAccuracy, trainConfMatrix = calculateAccuracy(normTrainData, target, weightsMatrix)
        testAccuracy, testConfMatrix = calculateAccuracy(normTestData, target, weightsMatrix)

        trainLogLoss = calculateLogLoss(normTrainData, target, weightsMatrix)
        testLogLoss = calculateLogLoss(normTestData, target, weightsMatrix)

        if trainLogLoss < lowestLogLoss:
            lowestLogLoss = trainLogLoss
            bestFold = n + 1
            bestLogLossList = logLossList

        # accuracy and recall calculations
        truePos = trainConfMatrix.at[1, 1]
        falseNeg = trainConfMatrix.at[0, 1]
        falsePos = trainConfMatrix.at[1, 0]
        trainPrecisionList.append(truePos / (truePos + falsePos))
        trainRecallList.append(truePos / (truePos + falseNeg))

        truePos = testConfMatrix.at[1, 1]
        falseNeg = testConfMatrix.at[0, 1]
        falsePos = testConfMatrix.at[1, 0]
        testPrecisionList.append(truePos / (truePos + falsePos))
        testRecallList.append(truePos / (truePos + falseNeg))

        trainAccuracyList.append(trainAccuracy)
        testAccuracyList.append(testAccuracy)
        trainLogLossValues.append(trainLogLoss)
        testLogLossValues.append(testLogLoss)

        # confusion matrix addition
        trainConfusionMatrix = pd.concat([trainConfusionMatrix, trainConfMatrix], sort=True).groupby(level=0).sum()
        testConfusionMatrix = pd.concat([testConfusionMatrix, testConfMatrix], sort=True).groupby(level=0).sum()

        if verbose:
            print("\nFold #", n + 1)
            print("Weights for normalized data using Gradient Descent:")
            print(weightsMatrix)
            print('Test Accuracy:', testAccuracy)
            print('Training Log Loss:', trainLogLoss)
            print('Testing Log Loss:', testLogLoss)
            print("---------------------------------------------------")

    # reslults coallated into a dataframe and returned

    mean = st.mean(trainAccuracyList)
    stdDev = st.stdev(trainAccuracyList)
    trainAccuracyList.append(mean)
    trainAccuracyList.append(stdDev)

    mean = st.mean(testAccuracyList)
    stdDev = st.stdev(testAccuracyList)
    testAccuracyList.append(mean)
    testAccuracyList.append(stdDev)

    mean = st.mean(trainPrecisionList)
    stdDev = st.stdev(trainPrecisionList)
    trainPrecisionList.append(mean)
    trainPrecisionList.append(stdDev)

    mean = st.mean(trainRecallList)
    stdDev = st.stdev(trainRecallList)
    trainRecallList.append(mean)
    trainRecallList.append(stdDev)

    mean = st.mean(testPrecisionList)
    stdDev = st.stdev(testPrecisionList)
    testPrecisionList.append(mean)
    testPrecisionList.append(stdDev)

    mean = st.mean(testRecallList)
    stdDev = st.stdev(testRecallList)
    testRecallList.append(mean)
    testRecallList.append(stdDev)

    mean = st.mean(trainLogLossValues)
    stdDev = st.stdev(trainLogLossValues)
    trainLogLossValues.append(mean)
    trainLogLossValues.append(stdDev)

    mean = st.mean(testLogLossValues)
    stdDev = st.stdev(testLogLossValues)
    testLogLossValues.append(mean)
    testLogLossValues.append(stdDev)

    resultsTable = pd.DataFrame()
    resultsTable["Fold #"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'Mean', 'Std Deviation']
    resultsTable["Train Accuracy"] = trainAccuracyList
    resultsTable["Train Precision"] = trainPrecisionList
    resultsTable["Train Recall"] = trainRecallList
    resultsTable["Train Log-Loss"] = trainLogLossValues
    resultsTable["Test Accuracy"] = testAccuracyList
    resultsTable["Test Precision"] = testPrecisionList
    resultsTable["Test Recall"] = testRecallList
    resultsTable["Test Log-Loss"] = testLogLossValues

    return resultsTable, trainConfusionMatrix, testConfusionMatrix, (bestFold, bestLogLossList)


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


def gradientDescent(originalData, target, learningRate, tolerance):
    '''
    Calculates the weights to assign to features using gradient descent. Also
    returns LogLoss list of each LogLoss value per iteration.
    
    :param originalData: pandas dataframe data
    :param target: target column name
    :param learningRate: learning rate to limit gradient descent jumps
    :param tolerance: tolerence threshold to determine when gradient descent
                        has converged
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

    priorLogLoss = calculateLogLoss(data, target, weightsMatrix)
    logLossList = [priorLogLoss]

    while not converged:
        # calculate predicted values using current weights
        predictedMatrix = np.dot(dataMatrix, weightsMatrix)
        predictedMatrix = sigmoid(predictedMatrix)

        col0 = np.array([dataMatrix[:, 0]]).T
        col1toN = dataMatrix[:, 1:]

        # gradient is maximizing log likelyhood by minimizing the negative log likelihood
        # from setting partial derivative of neg log-loss with respect to weights equal to 0.
        # gradient is regularized on all non-constant features (so not w0) to prevent overfitting
        grad0 = col0.T.dot(predictedMatrix - targetMatrix)
        grad1toN = (col1toN.T.dot(predictedMatrix - targetMatrix)) - (((len(col1toN)*.25)/ len(col1toN)) * (weightsMatrix[1:]))
        gradient = np.insert(grad1toN, 0, grad0[0], 0)

#         print(gradient)
        newWeights = (weightsMatrix - (learningRate * gradient))
        weightsMatrix = newWeights

        currentLogLoss = calculateLogLoss(data, target, weightsMatrix)

        if (abs(priorLogLoss - currentLogLoss) < tolerance):
            converged = True

        priorLogLoss = currentLogLoss
        logLossList.append(priorLogLoss)

        count += 1

        # MAXIMUM of 1000 iterations
        if count >= 1000:
            converged = True

    return weightsMatrix, logLossList


def calculateLogLoss(data, target, weightsMatrix):
    '''
    After gradient descent, this function can be used to determine 
    the Log loss indicating how much error the model has over the data and its 
    predictions.
     
    :param data: testing data dataframe object
    :param target: target column name
    :param weightsMatrix: weights used to predict class based on features
    '''

    targetCol = data[target].values.tolist()
    targetMatrix = np.array([[n] for n in targetCol])

    dataMatrix = np.array(data.drop(columns=[target]))

    predictedMatrix = np.dot(dataMatrix, weightsMatrix)
    predictedMatrix = sigmoid(predictedMatrix)

    logLoss = -targetMatrix * np.log(predictedMatrix) - ((1 - targetMatrix) * np.log(1 - predictedMatrix))

    logLoss = logLoss.sum() / (len(data))

    return logLoss


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
            prediction = 0
        else:
            prediction = 1
        actual = targetMatrix[n][0]

#         print("prediction:", prediction, "actual:", actual, "  ")

        confusionMatrix.at[(prediction), (actual)] = confusionMatrix.at[(prediction), (actual)] + 1

        if int(prediction) == int(actual):
            count += 1

    acc = count / len(predictedMatrix)

    return acc, confusionMatrix


def sigmoid(predictions):
    '''
    Normalizes prediction probabilities to be between 0 an 1
    
    :param predictions: predictions made during gradient descent (or after for testing)
    '''
    sig = 1 / (1 + np.exp(-predictions))

    # fixes an unexpected rounding error
    sig[sig == 1] = 0.9999999999999999
    sig[sig == 0] = 0.00000000000000000000000001

    return sig

