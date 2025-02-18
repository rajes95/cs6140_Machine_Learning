'''
CS 6140 Machine Learning - Assignment 03

Problem 2 - Naive Bayes Document Classification - Document Vocabulary Dataset
Multivariate Bernoulli Model AND Multinomial Event Model

@author: Rajesh Sakhamuru
@version: 6/30/2020
'''

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt


def trainBernoulliModel(dataMatrix, labelData):
    '''
    Given vocabXcategory data and labels, trains bernoulli model
    probabilities.
    
    :param dataMatrix: pandas dataframe
    :param labelData: pandas dataframe of which document is which category
    '''

    classProbs = (labelData[0].value_counts()) / len(labelData)
    classCounts = labelData[0].value_counts()

    classProbs.sort_index(axis=0, inplace=True)  # pi
    classCounts.sort_index(axis=0, inplace=True)

    dataMatrix[dataMatrix > 0] = 1

    dataMatrix['cat'] = labelData[0]

    catWordCounts = dataMatrix.groupby('cat').sum()
    dataMatrix = dataMatrix.drop('cat', axis=1)

    # simple smoothing
    countsPlus1 = catWordCounts + 1
    classCountsPlus2 = classCounts + 2

    probWordGivenClass = countsPlus1.div(classCountsPlus2, axis=0)  # weights

    return classProbs, probWordGivenClass


def trainMultinomialModel(dataMatrix, labelData):
    '''
    Given vocabXcategory data and labels, trains multinomial model
    probabilities.
    
    :param dataMatrix: pandas dataframe
    :param labelData: pandas dataframe of which document is which category
    '''

    classProbs = (labelData[0].value_counts()) / len(labelData)
    classProbs.sort_index(axis=0, inplace=True)  # pi

    dataMatrix['cat'] = labelData[0]

    catWordCounts = dataMatrix.groupby('cat').sum()
    dataMatrix = dataMatrix.drop('cat', axis=1)

    classCounts = catWordCounts.sum(axis=1)

    # simple smoothing
    countsPlus1 = catWordCounts + 1
    classCountsPlus2 = classCounts + 2

    probWordGivenClass = countsPlus1.div(classCountsPlus2, axis=0)  # weights

    return classProbs, probWordGivenClass


def predict(row, classProbs, probWordGivenClass, vocab):
    '''
    predict class based on row given and weights
    
    :param row: row testing model
    :param classProbs: percent of times class occurs in training data
    :param probWordGivenClass: probability of word occuring given the class
    :param vocab: list of testMatrix Columns
    '''
    rowWeights = probWordGivenClass.copy()
    for w in vocab:
        if row[w] == 0:
            rowWeights[w] = 1 - rowWeights[w]

    predictWeights = []
    for label, weights in rowWeights.iterrows():
        # not using this calculation because causes underflow (very low decimals that can be misread as 0)
#         classProb = classProbs[label]
#         numer = classProb * np.prod(weights)

        # gives same class result as above calculation without underflow
        classProb = np.log(classProbs[label])
        numer = classProb + np.sum(np.log(weights))

        predictWeights.append(numer)

    # expected normalization, but log normalization works better
#     total = np.sum(predictWeights)
#     for n in range(len(predictWeights)):
#         predictWeights[n] = predictWeights[n] / total

    bestClass = list.index(predictWeights, max(predictWeights)) + 1

    return bestClass


def getAccuracy(testMatrix, testLabels, classProbs, probWordGivenClass, bernoulli=False):
    '''
    given testing data, accuracy and confusion matrix are generated using the probabilites
    generated by the model. Bernoulli model should sent bernoulli=True in parameters.
    
    :param testMatrix: number of occurences of each vocab word per class
    :param testLabels: actual label of each document
    :param classProbs: percent of times class occurs in training data
    :param probWordGivenClass: probability of word occuring given the class
    :param bernoulli: boolean to indicate bernoulli model was used to generate probabilities
    '''

    if bernoulli:
        testMatrix[testMatrix > 0] = 1

    vocab = testMatrix.columns.tolist()

    targetCats = classProbs.index.tolist()
    confusionDict = {}
    for cat in targetCats:
        confusionDict[cat] = [0] * len(targetCats)
    confusionMatrix = pd.DataFrame(data=confusionDict, index=targetCats)

    count = 0
    counter = 0
    for index, row in testMatrix.iterrows():

        actual = testLabels[0][index - 1]
        prediction = predict(row, classProbs, probWordGivenClass, vocab)

#         print("prediction:", prediction, "actual:", actual)

        if int(prediction) == int(actual):
            count += 1

        confusionMatrix.at[(prediction), (actual)] = confusionMatrix.at[(prediction), (actual)] + 1
        counter += 1
        if counter % 100 == 0:
            print("rows tested:", counter)
            print("Accuracy so far:", count / counter)
            print(confusionMatrix)

    accuracy = count / len(testLabels)
    return accuracy, confusionMatrix


def generateWordFrequencyList(data):
    '''
    processes training data and generates the number of times each word occurs
    saves it as a CSV file
    
    :param data: given training data from CSV
    '''
    uniqueWords = data[1].unique().tolist()

    freqList = pd.DataFrame(columns=['Count'], index=uniqueWords)

    freqList[:] = 0

    lenData = len(data)

    for n in range(lenData):
        freqList['Count'][data[1][n]] += data[2][n]

    freqList.sort_values(by=['Count'], inplace=True, ascending=False)

    freqList.to_csv('datasets/20NG_data/train_vocabFreqList.csv')

    return


def selectVocab(vocabFreqList, limit):
    '''
    selects the top number of words based on limit
    
    :param vocabFreqList: vocab frequency list
    :param limit: number of words to select
    '''
    return vocabFreqList[:limit]


def generateLabelMatrix(labelData):
    '''
    generates label matrix which assigns binary label to each document
    :param labelData: label data given taken from CSV
    '''
    return pd.get_dummies(labelData[0])


def generateLabelAndDataMatrices(originalData, labelData, vocabList, path='datasets/20NG_data/csvFile.csv'):
    '''
    Generates Label and Data Matrices based on original data given. Will take a LONG time 
    for larger vocabList lengths. Saves CSV to file so it is not generated each time.
    
    :param originalData: given training data from CSV
    :param labelData: given label data from CSV
    :param vocabList: vocab list of specified length
    :param path: path to save data matrix (with a default value that is usually changed)
    '''
    labelMatrix = pd.get_dummies(labelData[0])
    print(labelMatrix)

    data = originalData.copy()

    wordList = vocabList['word'].tolist()

    print(wordList)
    newData = pd.DataFrame()

    count = 0

    for _, row in data.iterrows():
        if row[1] in wordList:
            newData = newData.append(row, ignore_index=True)

        # can be uncommented for testing to see that this function works properly without wasting time
#         if count > 10000:
#             break

        count += 1

    data = newData
    print(data)

    uniqueDocs = data[0].unique().tolist()
#     uniqueWords = data[1].unique().tolist()
    uniqueWords = wordList

    dataMatrix = pd.DataFrame(columns=uniqueWords, index=uniqueDocs)

    dataMatrix[:] = 0

    lenData = len(data)

    for n in range(lenData):
        dataMatrix[data[1][n]][data[0][n]] = data[2][n]

    print(dataMatrix)
    dataMatrix.sort_index(axis=1, inplace=True)
    dataMatrix.to_csv(path)

    return labelMatrix, dataMatrix, path


def recallAndPrecision(confMatrix):
    '''
    Returns recall and precision per class as dataframes
    
    :param confMatrix: confusion matrix of model testing results
    '''

    truePos = pd.DataFrame(pd.Series(np.diag(confMatrix), index=[confMatrix.index]))
    colSums = pd.DataFrame(confMatrix.sum(axis=1))
    rowSums = pd.DataFrame(confMatrix.sum(axis=0))

    recallList = pd.DataFrame(truePos[0].tolist()) / pd.DataFrame(colSums[0].tolist())
    precisionList = pd.DataFrame(truePos[0].tolist()) / pd.DataFrame(rowSums[0].tolist())

#     print(recallList)
#     print(precisionList)

    return recallList, precisionList


def groupedBarPlot(list1, list2, classes, title, ylabel):
    '''
    plots grouped bar plot of 2 lists, over the classes
    
    :param list1: list 1 of data
    :param list2: list 2 of data
    :param classes: classes of x axis
    :param title: graph title
    :param ylabel: y axis label
    '''

    labels = classes
    bern = list1
    mult = list2

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, bern, width, label='Multivar Bernoulli')
    ax.bar(x + width / 2, mult, width, label='Multinomial')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()


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


def accPrecRecVSVocabGraphs():
    '''
    Accuracy, Precision and Recall data has been manually entered over 7 different
    vocab sizes to be plotted by this function.
    '''

    xAxis = [100, 500, 1000, 2500, 5000, 7500, 10000]

    multivarAcc = [.180, .364, .463, .550, .596, .613, .622]
    multivarPrec = [.177, .359, .456, .543, .588, .604, .613]
    multivarRec = [.178, .423, .523, .618, .669, .691, .704]
    multinomAcc = [.228, .500, .614, .705, .743, .755, .763]
    multinomPrec = [.223, .494, .607, .698, .737, .749, .757]
    multinomRec = [.211, .508, .620, .711, .751, .764, .772]

    plot(xAxis, multivarAcc, "Multivar Bernoulli Accuracy vs Vocab Size", "Vocab Size", "Accuracy")
    plot(xAxis, multivarPrec, "Multivar Bernoulli Precision vs Vocab Size", "Vocab Size", "Precision")
    plot(xAxis, multivarRec, "Multivar Bernoulli Recall vs Vocab Size", "Vocab Size", "Recall")
    plot(xAxis, multinomAcc, "Multinomial Event Accuracy vs Vocab Size", "Vocab Size", "Accuracy")
    plot(xAxis, multinomPrec, "Multinomial Event Precision vs Vocab Size", "Vocab Size", "Precision")
    plot(xAxis, multinomRec, "Multinomial Event Recall vs Vocab Size", "Vocab Size", "Recall")


def main():
    # data visualization options for console output
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.expand_frame_repr', False)

    # default vocab limit is 100
    limit = 100

    # vocab size is set to the command line argument, otherwise defaults to 100
    try:
        print("VOCABULARY SIZE:", sys.argv[1])
        limit = int(sys.argv[1])
    except:
        print("\n------------------------------------------\nERROR: VOCAB SIZE NOT PASSED AS COMMAND LINE ARGUMENT")
        print("Possible arguments are: 100, 500, 1000, 2500, 5000, 7500 or 10000")
        print("try 'python3 documentNaiveBayes.py 100'")
        print("Defaulting to vocab list length: 100\n------------------------------------------\n")

    colNames = range(3)

    # import data file
    try:
        trainData = pd.read_csv("datasets/20NG_data/train_data.csv", delimiter=' ', names=colNames)
        trainLabels = pd.read_csv("datasets/20NG_data/train_label.csv", names=[0])
        vocabFreqList = pd.read_csv("datasets/20NG_data/train_vocabFreqList.csv", names=['word', 'count'])
#         testData = pd.read_csv("datasets/20NG_data/test_data.csv", delimiter=' ', names=colNames)
        testLabels = pd.read_csv("datasets/20NG_data/test_label.csv", names=[0])
    except:
        print("File not found")

    docNums = range(1, len(trainLabels) + 1)

    trainLabels.index = docNums

    print("Train Data:")
    print(trainData, "\n")
    print("Label Data:")
    print(trainLabels)

    vocabList = selectVocab(vocabFreqList, limit)
    print("Vocab Frequency List:")
    print(vocabList)

    # The below 3 lines of code generate the VOCAB FREQUENCY LIST and DATAMATRIX CSVs which are saved to the 20NG_data folder
    # They are already saved and therefore commented out and the files are loaded in directly.

#     generateWordFrequencyList(trainData)
#     labelMatrix, dataMatrix, dataMatrixPath = generateLabelAndDataMatrices(trainData, trainLabels, vocabList, path = 'datasets/20NG_data/top' + str(len(vocabList)) + 'DataMatrix.csv')
#     labelMatrix, dataMatrix, dataMatrixPath = generateLabelAndDataMatrices(testData, testLabels, vocabList, path = 'datasets/20NG_data/testTop' + str(len(vocabList)) + 'DataMatrix.csv')

    try:
        dataMatrix = pd.read_csv('datasets/20NG_data/top' + str(limit) + 'DataMatrix.csv', header=0, index_col=0)
        dataMatrix.index = dataMatrix.index.astype(int)
        columns = list(dataMatrix.columns.tolist())
        for n in range(len(columns)):
            columns[n] = int(float(columns[n]))
        dataMatrix.columns = columns

        testMatrix = pd.read_csv('datasets/20NG_data/testTop' + str(limit) + 'DataMatrix.csv', header=0, index_col=0)
        testMatrix.index = testMatrix.index.astype(int)
        columns = list(testMatrix.columns.tolist())
        for n in range(len(columns)):
            columns[n] = int(float(columns[n]))
        testMatrix.columns = columns
    except:
        print("File not found")

    print(dataMatrix)  # vocab x document
    print(testMatrix)  # used in getAccuracy()

    # trains bernoulli and multinomial models (generates probabilities as matrices)
    classProbsB, probWordGivenClassB = trainBernoulliModel(dataMatrix, trainLabels)
    classProbs, probWordGivenClass = trainMultinomialModel(dataMatrix, trainLabels)

    print(classProbs)
    print(probWordGivenClass)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # Calculates Accuracy and generates confusion matrix using weights and test datamatrix
    accuracyB, confusionMatrixB = getAccuracy(testMatrix, testLabels, classProbsB, probWordGivenClassB, bernoulli=True)
    accuracy, confusionMatrix = getAccuracy(testMatrix, testLabels, classProbs, probWordGivenClass)

    print("Multivar Bernoulli Confusion Matrix")
    print(confusionMatrixB)
    print("Multinomial Confusion Matrix")
    print(confusionMatrix)

    recallListB, precisionListB = recallAndPrecision(confusionMatrixB)
    recallList, precisionList = recallAndPrecision(confusionMatrix)

    print("Multivar Bernoulli Accuracy:", accuracyB)
    print("Multivar Bernoulli Avg Precision:", np.average(precisionListB[0].tolist()))
    print("Multivar Bernoulli Avg Recall:", np.average(recallListB[0].tolist()))
    print("Multinomial Accuracy:", accuracy)
    print("Multinomial Avg Precision:", np.average(precisionList[0].tolist()))
    print("Multinomial Avg Recall:", np.average(recallList[0].tolist()))

    bernPres = precisionListB[0].tolist()
    multPres = precisionList[0].tolist()
    bernRec = recallListB[0].tolist()
    multRec = recallList[0].tolist()
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

    groupedBarPlot(bernPres, multPres, classes, "Multivar Bernoulli vs Multinomial Precision per Class", "Precision")
    groupedBarPlot(bernRec, multRec, classes, "Multivar Bernoulli vs Multinomial Recall per Class", "Recall")


main()
