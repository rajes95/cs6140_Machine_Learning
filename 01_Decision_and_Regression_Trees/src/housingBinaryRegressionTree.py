'''
CS 6140 Machine Learning - Assignment 01

Problem 6 - Regression Trees - Housing Dataset

@author: Rajesh Sakhamuru
@version: 5/21/2020
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

    # list of SSEs over the folds
    SSEValues = []

    # 10-fold cross validation of growTree model
    for n in range(10):
        print("Fold #", n)
        # split into training and testing data
        testData = dataShuffled[(n * oneTenthRows):oneTenthRows * (n + 1)]
        trainData = dataShuffled.drop(dataShuffled.index[(n * oneTenthRows):oneTenthRows * (n + 1)])

        # calculate minimum leaf node size
        minLeaf = math.ceil(nMin * len(trainData))

        # grow the decision tree and return it as a dictionary object
        tr = growBinaryRegressionTree(trainData, target, colNames, minLeaf)

        # encode the grown decision tree dictionary as a function to file for use
        # in prediction
        encodeBinaryRegressionTree(tr, colNames, target, location=False)

        # calculate fold's SSE and save it to SSE list
        trSSE = calculateTreeSSE(testData, target)
        print("Fold SSE:", trSSE)
        SSEValues.append(trSSE)

    return SSEValues


def calculateTreeSSE(testData, target):
    '''
    After construction of the decision tree, this function can be used to determine 
    the SSE indicating how much error the tree has over the test data and its 
    predictions.
    
    :param testData: testing data dataframe object
    :param target: target column name
    '''

    targetCol = testData[target].values.tolist()

    testData = testData.drop(columns=[target])
    testRowsList = testData.values.tolist()

    trSSE = 0

    # calculate squared errors for each row in testing data and sum them for the SSE
    for n in range(len(testRowsList)):
        row = testRowsList[n]

        for m in range(len(row)):
            if type(row[m]) == int:
                row[m] = str(float(row[m]))
            else:
                row[m] = str(row[m])

        prediction = str(dt.predict(row))
        actual = str(targetCol[n])
#         print(row, "prediction:", prediction, "actual:", actual, "  ")
        prediction = float(prediction)
        actual = float(actual)

        trSSE += (prediction - actual) * (prediction - actual)

    return trSSE


def encodeBinaryRegressionTree(tree, colNames, target, location=True):
    '''
    Encodes the first line of the .py file used to classify data. Writes the 'def'
    line. then calls a helper function to encode the tree as if/elif statements.
    
    :param tree: dictionary tree object
    :param colNames: list of column names
    :param target: target column
    :param location: flag for printing the location of the classifier.py file
    '''

    try:
        file = open("output/classifier.py", "a+")
    except:
        print("File not found")
        exit(-1)

    file.truncate(0)
    file.writelines("def classify(obj):   # Columns -> ")

    colDict = {}
    count = 0
    for name in colNames:
        if name == target:
            continue

        colDict[name] = count

        if count != len(colNames) - 2:
            file.writelines("obj[" + str(count) + "]: " + str(name) + ", ")
        else:
            file.writelines("obj[" + str(count) + "]: " + str(name))

        count += 1

    file.writelines("\n")

    encodeTreeToFile(tree, colDict, file)

    # optional verbose option with 'location' flag
    if location:
        print("Classification tree encoded to 'classify(obj)' function in " + \
          "./output/classifier.py")

    file.close()


def encodeTreeToFile(tree, colDict, file, key=None, depth=0):
    '''
    encodes the dictionary tree to the file as executable code for classification
    of test data.
    
    :param tree: dictionary tree object
    :param colDict: dictionary of column names as keys and their number (based on order) as items 
    :param file: .py file being written to
    :param key: dictionary key to the next recursion of the dict
    :param depth: keeping track of depth of tree for correct tabs and spaces
    '''
    edges = []

    if key == None:
        key = list(tree.keys())[0]

    col = key[0]
    split = key[1]

    # edges of immediate node/key
    edges += list(tree[key].keys())

    # if or elif depending on if it's the first or later branch
    ifOrElif = "if"
    # inequality symbol varies depending on if it's the first or later branch
    inequality = "<"

    for e in edges:
        # child is either a single value (leaf node) or a dict type which is not
        # a leaf node and needs to be recursively encoded
        child = tree[key][e]

        # fixes an edgecase where '< 0' is not possible so it becomes '<= 0'
        if split == 0:
            if inequality == "<":
                inequality = "<="

        if type(child) == dict:
            line = (((depth + 1) * "    ") + ifOrElif + " float(obj[" + str(colDict[col])
                  +"]) " + inequality + " " + str(split) + ":")
            file.writelines(line + "\n")
            encodeTreeToFile(child, colDict, file, key=list(child.keys())[0], depth=depth + 1)
        else:
            line = (((depth + 1) * "    ") + ifOrElif + " float(obj[" + str(colDict[col])
                  +"]) " + inequality + " " + str(split) + ":\n" + ((depth + 2) * "    ") + "return '" + str(child) + "'")
            file.writelines(line + "\n")
        ifOrElif = "elif"
        if inequality == "<":
            inequality = ">="
        else:
            inequality = ">"


def growBinaryRegressionTree(currentData, target, colNames, minLeaf, decTree=None):
    '''
    Grows the regression tree minimizing SSE values to prioritize nodes with 
    similar target values.
    
    :param currentData: dataset used to train tree
    :param target: target column name
    :param colNames: column name list
    :param minLeaf: minimum leaf size based on threshold (nMin)
    :param decTree: tree being built as dictionary, grown recursively
    '''

    # calculate best feature and value to split it at
    feat, bestSplit = bestRegressionFeat(currentData, target, colNames)

    # tupleized to be stored as an object in the decTree dictionary
    nodeName = (feat, bestSplit)

    # if decision tree is empty, initialize decision tree
    if decTree == None:
        decTree = {}
        decTree[nodeName] = {}

    branch1 = currentData[currentData[feat] < bestSplit]
    branch2 = currentData[currentData[feat] >= bestSplit]

    flag = True

    # fixes edgecase where the best split is at 0 and there are other data with 0
    # in the feature column
    if branch1.empty or branch2.empty:
        flag = False
        branch1 = currentData[currentData[feat] <= bestSplit]
        branch2 = currentData[currentData[feat] > bestSplit]

    branches = [branch1, branch2]

    # each subtable is its own branch.
    for subTable in branches:

        subTableLen = len(subTable)
        leafFlag = True

        # fixes edgecase where all column data is the same with different targets
        for col in colNames:
            if col is target:
                continue
            if subTable[col].value_counts().max() != len(subTable):
                leafFlag = False
                break
        # leaf node
        if len(subTable[target].unique()) == 1 or subTableLen <= minLeaf or leafFlag:

            leaf = predictedValue(subTable, target)

            if flag:
                if float(subTable[feat].max()) < bestSplit:
                    decTree[nodeName][True] = leaf
                else:
                    decTree[nodeName][False] = leaf
            else:
                if float(subTable[feat].max()) <= bestSplit:
                    decTree[nodeName][True] = leaf
                else:
                    decTree[nodeName][False] = leaf

        # not a leaf node
        else:
            # recursively call growBinaryRegressionTree to build rest of tree if
            # it is a non-leaf node
            tr = growBinaryRegressionTree(subTable, target, colNames, minLeaf)

            if flag:
                if float(subTable[feat].max()) < bestSplit:
                    decTree[nodeName][True] = tr
                else:
                    decTree[nodeName][False] = tr
            else:
                if float(subTable[feat].max()) <= bestSplit:
                    decTree[nodeName][True] = tr
                else:
                    decTree[nodeName][False] = tr

    return decTree


def bestRegressionFeat(data, target, colNames):
    '''
    Given a dataframe, it finds the best feature and value of that feature to 
    split the data by, for a binary regression tree.
    
    :param data: pandas dataframe object
    :param target: target column name
    :param colNames: list of column names
    '''

    dataCopy = data.copy()

    cols = colNames.copy()
    cols.remove(target)

    lowestSSE = math.inf
    bestCol = ''
    bestSplit = 0

    # loops through list of columns
    for col in cols:

        # ignores homogenous columns that won't help with the split
        if len(dataCopy[col].unique()) == 1:
            continue

        SSE, splitVal = bestSSEDropInCol(dataCopy, target, col)

        if SSE < lowestSSE:
            lowestSSE = SSE
            bestCol = col
            bestSplit = splitVal

    return bestCol, bestSplit


def bestSSEDropInCol(data, target, col):
    '''
    Returns the best SSE value possible and the split location that allows that 
    for a single column.
    
    :param data: pandas dataframe object 
    :param target: target column name
    :param col: column where best SSE and split is being looked for
    '''

    dataCopy = data.copy()

    dataLen = len(dataCopy)

    sortedByCol = dataCopy.sort_values(by=[col])
    sortedByCol.reset_index(drop=True, inplace=True)

    bestSplit = 0
    lowSSE = math.inf

    # Tries every midpoint and calculates the SSE for the split and picks the
    # lowest SSE
    for n in range(dataLen - 1):

        leaf1 = sortedByCol[:n + 1]
        leaf2 = sortedByCol[n + 1:]

        split = (leaf1.at[n, col] + leaf2.at[n + 1, col]) / 2

        leaf1SSE = sumSqrError(leaf1, target)
        leaf2SSE = sumSqrError(leaf2, target)

        SSE = leaf1SSE + leaf2SSE

        if SSE < lowSSE:
            lowSSE = SSE
            bestSplit = split

    return lowSSE, bestSplit


def sumSqrError(data, target):
    '''
    Calculates the SSE of the data in relation to the mean target value.
    
    :param data: dataframe
    :param target: target column name
    '''
    val = predictedValue(data, target)

    targetCol = data[target].copy()
    targetCol = (targetCol - val)
    targetCol = targetCol * targetCol
    sumErrors = targetCol.sum()

    return sumErrors


def predictedValue(data, target):
    '''
    The predicted value of a node which is simply the average value of the target column
    at that node.
    
    :param data: data as a dataframe object
    :param target: target column name
    '''
    val = data[target].mean()
    return val


def main():
    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    # column names, nMin/threshold proportions and target column name initialized
    colNames = []
    for i in range(14):
        colNames.append(str(i))

    nMin = [0.05, 0.10, 0.15, 0.20]
    target = colNames[-1]

    # import data file
    try:
        housingData = pd.read_csv("datasets/housing.csv", names=colNames)
    except:
        print("File not found")

    # Using full dataset for the sake of testing will take way too long
    # (took a long time on my machine for full ten-fold crossvalidation) so you 
    # can use a smaller sample to verify that the code is working as intended if
    # you want.
    housingData = housingData.sample(150).reset_index(drop=True)

    print("Original Data:")
    print(housingData, "\n")

    # normalize all data before going into tenFoldCrossValidation
    normData = dt.normalizeData(housingData, colNames, target)
    print("Normalized Data:")
    print(normData)

    print("SSE of whole dataset without tree", sumSqrError(normData, target))

    print("\nClassification trees will be encoded to 'classify(obj)' function in " + \
          "./output/classifier.py\n")

    # for each nMin value perform nMin cross validation and return accuracy and
    # confusion matrix values
    for n in nMin:
        print("Doing 10-fold cross-validation for nMin=" + str(n) + "... (please wait, this can take a long time)\n")
        SSEValues = tenFoldCrossValidation(normData, colNames, target, n)
        print("Total SSE:", sum(SSEValues))
        print("Average Regression Tree SSE accross the Folds:", sum(SSEValues) / len(SSEValues))
        print("SSE Standard Deviation:", st.pstdev(SSEValues))
        print()


main()
