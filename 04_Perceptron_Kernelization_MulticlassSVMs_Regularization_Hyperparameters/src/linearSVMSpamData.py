'''
CS 6140 Machine Learning - Assignment 04

Problem 3 - Determining Model Hyper-parameters - Spambase Dataset Linear Kernel

@author: Rajesh Sakhamuru
@version: 7/20/2020
'''

import pandas as pd
import statistics as st

from sklearn.svm import SVC
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


def kFoldCrossValidation(data, kFolds=10):
    '''
    k fold cross validation SVM Linear kernel
    :param data: data as pandas dataframe
    :param kFolds: number of folds
    '''
    dataShuffled = data.sample(frac=1).reset_index(drop=True)

    colNames = getColNames(data)
    target = colNames[-1]

    numRows = len(dataShuffled)
    oneKthRows = round(numRows / kFolds)

    resultsTable = pd.DataFrame(columns=["OuterFold", "TrainAccuracy", \
            "TrainPrecision", "TrainRecall", "TestAccuracy", "TestPrecision", "TestRecall"])

    for k in range(kFolds):
        print("Fold #", k + 1, "...")
        testDataK = dataShuffled[(k * oneKthRows):oneKthRows * (k + 1)]
        trainDataK = dataShuffled.drop(dataShuffled.index[(k * oneKthRows):oneKthRows * (k + 1)])
        testDataK = testDataK.reset_index(drop=True)
        trainDataK = trainDataK.reset_index(drop=True)

        x_train = trainDataK.drop(target, axis=1)
        y_train = trainDataK[target]

        x_test = testDataK.drop(target, axis=1)
        y_test = testDataK[target]

        # gaussian kernel
        classifier = SVC(kernel='linear')

        # Train model
        classifier.fit(x_train, y_train)

        # TESTING model
        y_pred = classifier.predict(x_test)
        y_pred_train = classifier.predict(x_train)

        acc = metrics.accuracy_score(y_test, y_pred)
        prec = metrics.precision_score(y_test, y_pred)
        rec = metrics.recall_score(y_test, y_pred)

        accTrain = metrics.accuracy_score(y_train, y_pred_train)
        precTrain = metrics.precision_score(y_train, y_pred_train)
        recTrain = metrics.recall_score(y_train, y_pred_train)

        resultsTable.loc[len(resultsTable)] = [k + 1, accTrain, precTrain, recTrain, acc, prec, rec]

    accListTr = resultsTable["TrainAccuracy"].tolist()
    precListTr = resultsTable["TrainPrecision"].tolist()
    recListTr = resultsTable["TrainRecall"].tolist()
    accList = resultsTable["TestAccuracy"].tolist()
    precList = resultsTable["TestPrecision"].tolist()
    recList = resultsTable["TestRecall"].tolist()

    resultsTable.loc[len(resultsTable)] = ["Avg", st.mean(accListTr), st.mean(precListTr), st.mean(recListTr)\
            , st.mean(accList), st.mean(precList), st.mean(recList)]
    resultsTable.loc[len(resultsTable)] = ["StdDev", st.stdev(accListTr), st.stdev(precListTr), st.stdev(recListTr)\
            , st.stdev(accList), st.stdev(precList), st.stdev(recList)]

    return resultsTable


def getColNames(data):
    '''
    Returns list of column names of data given
    :param data: pandas dataframe
    '''
    return list(data.columns)


def main():
    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    # import data file

    colNames = range(58)
    try:
        spamData = pd.read_csv("datasets/spambase.csv", names=colNames)
    except:
        print("File not found")

    print(spamData)

    results = kFoldCrossValidation(spamData)

    pd.set_option('display.max_rows', None)

    print(results)

'''



































'''

main()
