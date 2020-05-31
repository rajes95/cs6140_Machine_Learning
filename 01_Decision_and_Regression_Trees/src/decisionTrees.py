'''
CS 6140 Machine Learning - Assignment 01

Problem 1.1 - Growing Decision Trees

@author: Rajesh Sakhamuru
@version: 5/14/2020
'''

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn import preprocessing


def categoricalToBinary(data, colNames, target):
    '''
    Converts categorical data to binary data by making each individual category
    in a feature its own binary column.
    
    :param data: categorical
    :param colNames: list of column names
    :param target: target column name
    '''

    binaryData = pd.DataFrame()

    colNames.remove(target)

    style = preprocessing.LabelBinarizer()

    count = 0

    # transorm each column into multiple columns binary, one for each category per
    # feature.
    for col in colNames:

        result = style.fit_transform(data[col])
        colClasses = list(style.classes_)

        if len(result[0]) != len(colClasses):
            colClasses.pop()

        for n in range(len(colClasses)):
            colClasses[n] = count
            count += 1

        df1 = pd.DataFrame(result, columns=colClasses)
        binaryData = pd.concat([df1, binaryData], axis=1, sort=False)

    targetCol = pd.DataFrame(data, columns=[target])
    targetCol = targetCol.rename(columns={target:count})

    binaryData = pd.concat([targetCol, binaryData], axis=1, sort=False)
    binaryData = binaryData.reindex(sorted(binaryData.columns), axis=1)

    return binaryData


def continuousToBinary(data, colNames, target):
    '''
    Dataframe with columns with continuous data are converted to binary columns
    at the threshold with the most information gain. 
    
    :param data: training data with continuous features
    :param colNames: list of column names that are continuous
    :param target: target column name
    '''
    
    cols = list(colNames)

    if target in cols:
        cols.remove(target)

    bestMids = []

    for col in cols:
        dataCopy = data.copy()
        sortedByCol = dataCopy.sort_values(by=[col])

        sortedColData = sortedByCol[col].tolist()

        best = 0
        bestMid = 0

        # at each midpoint between consecutive data in each column, 
        # the information gain is calculated, and the best mid is picked.
        for n in range(len(sortedColData) - 1):
            dataCopy = data.copy()
            mid = (sortedColData[n] + sortedColData[n + 1]) / 2

            dataCopy.loc[dataCopy[col] < mid, col] = 0
            dataCopy.loc[dataCopy[col] >= mid, col] = 1

            gain = infoGain(dataCopy, target, col, colNames)
            entropy.cache_clear()
            if gain > best:
                best = gain
                bestMid = mid

        bestMids.append(bestMid)
    
    # once best mids are picked the training data is fully converted into 
    # binary features and returned
    binaryData = data.copy()
    count = 0
    midCount = 0
    for col in cols:
        for i, _ in binaryData.iterrows():
            boolVal = ((binaryData.at[i, col]) < bestMids[midCount])
            if boolVal:
                binaryData.at[i, col] = str(int(count))
            else:
                binaryData.at[i, col] = str(int(count + 1))

        count += 2
        midCount += 1

    return binaryData, bestMids


def testRowsToBinary(data, bestMids, colNames, target):
    '''
    Using the best split points calculated in coninuousToBinary(), the test data
    is converted to binary as well.
    
    :param data: test data dataframe
    :param bestMids: list of binary split values calculated from training data.
    :param colNames: list of column names
    :param target: target column name
    '''

    cols = list(colNames)

    if target in cols:
        cols.remove(target)

    binaryData = data.copy()
    count = 0
    midCount = 0
    for col in cols:
        for i, _ in binaryData.iterrows():
            boolVal = ((binaryData.at[i, col]) < bestMids[midCount])
            if boolVal:
                binaryData.at[i, col] = str(int(count))
            else:
                binaryData.at[i, col] = str(int(count + 1))
        count += 2
        midCount += 1

    return binaryData


def normalizeData(data, colNames, target):
    '''
    Normalize each column continuous data to between 0 and 1
    
    :param data: dataframe with data
    :param colNames: list of column names
    :param target: name of target column
    '''

    featureNames = colNames.copy()

    featureNames.remove(target)

    features = pd.DataFrame(data.drop(columns=[target]), columns=featureNames)

    targetCol = pd.DataFrame(data, columns=[target])
    
    # using pre-built normalization function
    dataArray = features.values
    min_max_scaler = preprocessing.MinMaxScaler()
    normalizedDataArr = min_max_scaler.fit_transform(dataArray)
    normalizedFeatures = pd.DataFrame(normalizedDataArr, columns=featureNames)

    targetCol.reset_index(drop=True, inplace=True)
    
    # re-attach target column
    normalizedFeatures[target] = targetCol

    return normalizedFeatures


def calculateAccuracy(testData, target, targetCats, matrix=True):
    '''
    calculates accuracy of classifier tree and creates a confusion matrix for 
    data visualization
    
    :param testData: test data in dataframe
    :param target: name of target column
    :param targetCats: unique categories in target column
    :param matrix: flag for whether or not we want matrix output or just accuracy
    '''

    targetCats.sort()

    targetCats = [str(i) for i in targetCats]

    # empty confusion matrix
    if matrix:

        confusionDict = {}
        for cat in targetCats:
            confusionDict[cat] = [0] * len(targetCats)

        confusionMatrix = pd.DataFrame(data=confusionDict, index=targetCats)


    targetCol = testData[target].values.tolist()
    testData = testData.drop(columns=[target])
    testRowsList = testData.values.tolist()

    count = 0
    for n in range(len(testRowsList)):
        row = testRowsList[n]
        
        # enforces uniformity in values in row
        for m in range(len(row)):
            if type(row[m]) == int:
                row[m] = str(float(row[m]))
            else:
                row[m] = str(row[m])
        
        # actual vs predicted value calculated and used to populate matrix and 
        # accuracy value
        prediction = str(predict(row))
        actual = str(targetCol[n])

#         print(row, "prediction:", prediction, "actual:", actual, "  ")
        if matrix:
            confusionMatrix.at[prediction, actual] = confusionMatrix.at[prediction, actual] + 1
        if prediction == actual:
            count += 1

    accuracy = count / len(testRowsList)
#     print("accuracy:", accuracy, "\n")
    accuracy = round(accuracy, 3)

    if matrix:
        return accuracy, confusionMatrix

    return accuracy


def predict(row):
    '''
    imports the classifier classify function to predict the outcome of a single
    test data row
    
    :param row: List of features for a specific data row.
    '''
    try:
        from output.classifier import classify
    except:
        print("'classify' function not found at 'output.classifier'")
        exit(-1)

    return classify(row)


def encode(tree, colNames, target, location=True):
    '''
    encodes the first line of the .py file used to classify data. Writes the 'def'
    line.
    
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
    file.writelines("def classify(obj):   # ")

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
    
    # edges of immediate node/key
    edges += list(tree[key].keys())

    # if or elif depending on if it's the first or later branch
    ifOrElif = "if"

    for e in edges:
        
        # child is either a single value (leaf node) or a dict type which is not
        # a leaf node and needs to be recursively encoded
        child = tree[key][e]
        eStr = str(e)
        try:
            eStr = str(float(e))
        except ValueError:
            pass

        if type(child) == dict:
            line = (((depth + 1) * "    ") + ifOrElif + " obj[" + str(colDict[key])
                  +"] == '" + eStr + "':")
            file.writelines(line + "\n")
            encodeTreeToFile(child, colDict, file, key=list(child.keys())[0], depth=depth + 1)
        else:
            line = (((depth + 1) * "    ") + ifOrElif + " obj[" + str(colDict[key])
                  +"] == '" + eStr + "':\n" + ((depth + 2) * "    ") + "return '" + str(child) + "'")
            file.writelines(line + "\n")
        ifOrElif = "elif"


def clearGrowTreeCache():
    '''
    Clears growTree cache which is used for memoization.
    '''
    growTree.cache_clear()


@lru_cache(maxsize=None)
def growTree(dataTuple, target, colsTuple, minLeaf, decTree=None):
    '''
    Grows the tree based on the feature with the best information gain
    
    :param dataTuple: dataset as tuple for memoization
    :param target: target column name
    :param colsTuple: tuple of column name list
    :param minLeaf: minimum leaf size based on threshold (nMin)
    :param decTree: tree being built as dictionary
    '''
    
    # tuples back to list and dataframe (because memoization) 
    colNames = list(colsTuple)
    currentData = pd.DataFrame(dataTuple, columns=colNames)
    
    # calculate best feature with maximum gain
    feat = bestGainFeature(currentData, target, colNames)

    # if decision tree is empty, initialize decision tree
    if decTree == None:
        decTree = {}
        decTree[feat] = {}
    
    # loop through each unique category in the column with best information gain
    uniqueColCats = currentData[feat].unique()

    for s in uniqueColCats:
        
        # subtable based on each unique category
        subTable = currentData.loc[currentData[feat] == s]
        subTableLen = len(subTable)

        # flags edgecase as leaf if columns are all identical, 
        # but have different target values
        leafFlag = True
        for col in colNames:
            if col is target:
                continue
            if subTable[col].value_counts().max() != len(subTable):
                leafFlag = False
                break
        # leaf node
        if len(subTable[target].unique()) == 1 or subTableLen <= minLeaf or leafFlag:
            leaf = (subTable[target].value_counts().idxmax())
            decTree[feat][s] = leaf
            
        # not leaf node
        else:
            # recursively call growTree to get the rest of the dictionary/tree built
            # at a non-leaf node
            subTableTuple = tuple(subTable.itertuples(index=False, name=None))
            colsTuple = tuple(colNames)
            tr = growTree(subTableTuple, target, colsTuple, minLeaf)
            decTree[feat][s] = tr

    return decTree


def bestGainFeature(data, target, colNames):
    '''
    Cycles through all the columns to see which one provides the most information gain
    
    :param data: dataframe
    :param target: target column
    :param colNames: list of column names
    '''
    bestGain = 0
    bestCol = ''

    for col in colNames:
        if col is target:
            continue

        gain = infoGain(data, target, col, colNames)

        if gain >= bestGain:
            bestGain = gain
            bestCol = col

    return bestCol


def infoGain(data, target, col, colNames):
    '''
    Calculates information gain for a specific column in relation to the target column
    Uses entropy to make calculation.
    
    :param data: dataframe being used
    :param target: target column name
    :param col: column having IG calculated for
    :param colNames: list of column names
    '''
    dataTuple = tuple(data.itertuples(index=False, name=None))
    colsTuple = tuple(colNames)
    entropyData = entropy(dataTuple, target, colsTuple)

    uniqueColCategories = data[col].unique()

    infGain = entropyData

    for s in uniqueColCategories:
        subTable = data.loc[data[col] == s]
        tableLen = len(subTable)
        totalEntries = len(data)
        ratio = tableLen / totalEntries

        subTableTuple = tuple(subTable.itertuples(index=False, name=None))
        colsTuple = tuple(colNames)
        entr = entropy(subTableTuple, target, colsTuple)

        infGain -= ratio * entr

    return infGain


def clearEntropyCache():
    '''
    Clears memoization cache for entropy.
    '''
    entropy.cache_clear()


@lru_cache(maxsize=None)
def entropy(dataTuple, col, colsTuple):
    '''
    Calculates entropy of a dataset in regards to a specific target column
   
    :param dataTuple: dataset as tuple for memoization
    :param col: target column name
    :param colsTuple: column names for dataset
    '''
    colNames = list(colsTuple)
    data = pd.DataFrame(dataTuple, columns=colNames)

    uniqueColCategories = data[col].unique()
    totalEntries = len(data)
    entropyS = 0
    for s in uniqueColCategories:
        subTable = data.loc[data[col] == s]
        tableLen = len(subTable)
        probability = (tableLen / totalEntries)
        entropyS -= probability * np.log2(probability)

    return (entropyS)

