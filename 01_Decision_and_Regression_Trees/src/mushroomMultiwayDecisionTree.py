'''
CS 6140 Machine Learning - Assignment 01

Problem 2.1a - Classification Trees With Categorical Features - Multiway Mushroom Dataset 

@author: Rajesh Sakhamuru
@version: 5/17/2020
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
        
        # split into training and testing data
        testData = dataShuffled[(n * oneTenthRows):oneTenthRows * (n + 1)]
        trainData = dataShuffled.drop(dataShuffled.index[(n * oneTenthRows):oneTenthRows * (n + 1)])
        
        # convert dataframes and lists to tuples so that growTree can be memoized
        trainDataTuple = tuple(trainData.itertuples(index=False, name=None))
        colsTuple = tuple(colNames)
        minLeaf = math.ceil(nMin * len(trainData))

        # grow the decision tree and return it as a dictionary object
        tr = dt.growTree(trainDataTuple, target, colsTuple, minLeaf)
        dt.clearEntropyCache()

        # encode the grown decision tree dictionary as a function to file for use
        # in prediction
        dt.encode(tr, colNames, target, location=False)

        # calculate accuracy and the fold's confusion matrix
        accu, confMatrix = dt.calculateAccuracy(testData, target, targetCats)
        confusionMatrix = pd.concat([confusionMatrix, confMatrix], sort=True).groupby(level=0).sum()
        accuracyValues.append(accu)

    dt.clearGrowTreeCache()

    return accuracyValues, confusionMatrix


def main():
    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    # column names, nMin/threshold proportions and target column name initialized
    colNames = []
    for i in range(22):
        colNames.append(str(i))
    nMin = [0.05, 0.10, 0.15]
    target = colNames[-1]
    
    # import data file
    try:
        mushData = pd.read_csv("datasets/mushroom.csv", names=colNames)
    except:
        print("File not found")

    print("Mushroom Data:")
    print(mushData, "\n")

    print("\nClassification trees will be encoded to 'classify(obj)' function in " + \
          "./output/classifier.py\n")

    # for each nMin value perform nMin cross validation and return accuracy and
    # confusion matrix values
    for n in nMin:
        print("Doing 10-fold cross-validation for nMin=" + str(n) + "...\n")
        accuracyArr, confMatrix = tenFoldCrossValidation(mushData, colNames, target, n)
        print("\nnMin:", n)
        print("Average Accuracy:", sum(accuracyArr) / len(accuracyArr))
        print("Standard Deviation:", st.pstdev(accuracyArr))
        print("Confusion Matrix: (Columns are Actual, Rows are Predicted)")
        print(confMatrix)
        print()


main()
