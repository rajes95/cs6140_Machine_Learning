'''
CS 6140 Machine Learning - Assignment 01

Problem 1.1b - Growing Decision Trees - Spambase Dataset

@author: Rajesh Sakhamuru
@version: 5/16/2020
'''

import math

import pandas as pd
import statistics as st

import decisionTrees as dt


def tenFoldCrossValidation(data, colNames, target, nMin):
    '''
    This function generates decision trees based on 10 unique folds of the 
    dataset provided. Then the accuracy values over the folds and the confusion 
    matrix are returned.
    
    :param data: dataframe with all training and testing data 
    :param colNames: list of column names
    :param target: target column name
    :param nMin: nMin threshold proportion value. 0 < nMin < 1
    '''
    # prevent changes to original data
    data = data.copy()

    # shuffle data
    dataShuffled = data.sample(frac=1).reset_index(drop=True)
    numRows = len(dataShuffled)
    oneTenthRows = int(numRows / 10)

    accuracyValues = []

    # empty confusion matrix to be filled in by dt.calculateAccuracy()'s return
    targetCats = data[target].unique()
    targetCats = [str(i) for i in targetCats]
    confusionDict = {}
    for cat in targetCats:
        confusionDict[cat] = [0] * len(targetCats)
    confusionMatrix = pd.DataFrame(data=confusionDict, index=targetCats)

    # 10-fold cross validation of growTree model
    for n in range(10):
        print("Fold #", n)
        testData = dataShuffled[(n * oneTenthRows):oneTenthRows * (n + 1)]
        trainData = dataShuffled.drop(dataShuffled.index[(n * oneTenthRows):oneTenthRows * (n + 1)])

        print("Normalizing Training and Test data...")
        testData = dt.normalizeData(testData, colNames, target)
        trainData = dt.normalizeData(trainData, colNames, target)

        print("Converting continuous data to binary...")
        binaryTrainData, bestMids = dt.continuousToBinary(trainData, colNames, target)
        binaryTestData = dt.testRowsToBinary(testData, bestMids, colNames, target)

        print("Growing decision tree...")
        trainDataTuple = tuple(binaryTrainData.itertuples(index=False, name=None))
        colsTuple = tuple(colNames)
        minLeaf = math.ceil(nMin * len(trainData))
        tr = dt.growTree(trainDataTuple, target, colsTuple, minLeaf)

        dt.clearEntropyCache()

        # encode the grown decision tree dictionary as a function to file for use
        # in prediction
        dt.encode(tr, colNames, target, location=False)

        # calculate accuracy and the fold's confusion matrix
        accu, confMatrix = dt.calculateAccuracy(binaryTestData, target, targetCats)
        confusionMatrix = pd.concat([confusionMatrix, confMatrix], sort=True).groupby(level=0).sum()
        print("Accuracy in fold #", n, ":", accu)
        accuracyValues.append(accu)

    dt.clearGrowTreeCache()

    return accuracyValues, confusionMatrix


def main():
    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    colNames = []
    for i in range(58):
        colNames.append(str(i))

    # column names, nMin/threshold proportions and target column name initialized
    nMin = [0.05, 0.10, 0.15, 0.20, 0.25]
    target = colNames[-1]

    # import data file
    try:
        spamData = pd.read_csv("datasets/spambase.csv", names=colNames)
    except:
        print("File not found")

    # Using full dataset for the sake of testing will take way too long
    # (took over 10 hours on my machine for full ten-fold crossvalidation) so you 
    # can use a smaller sample to verify that the code is working as intended.
    spamData = spamData.sample(120).reset_index(drop=True)

    print("Spambase Data:")
    print(spamData, "\n")

    # normalize all data before going into tenFoldCrossValidation
    normData = dt.normalizeData(spamData, colNames, target)
    print("Normalized Data:")
    print(normData)

    print("\nClassification trees will be encoded to 'classify(obj)' function in " + \
          "./output/classifier.py\n")

    # for each nMin value perform nMin cross validation and return accuracy and
    # confusion matrix values
    for n in nMin:
        print("Doing 10-fold cross-validation for nMin=" + str(n) + "... (please wait... can take a long time)\n")
        accuracyArr, confMatrix = tenFoldCrossValidation(normData, colNames, target, n)
        print("\nnMin:", n)
        print("Average Accuracy:", sum(accuracyArr) / len(accuracyArr))
        print("Standard Deviation:", st.pstdev(accuracyArr))
        print("Confusion Matrix: (Columns are Actual, Rows are Predicted)")
        print(confMatrix)
        print()


main()
