'''
CS 6140 Machine Learning - Assignment 04

Problem 3 - Determining Model Hyper-parameters - Breast Cancer Dataset RBF/Gaussian Kernel
Hyperparameters chosen by optimizing for AUC

@author: Rajesh Sakhamuru
@version: 7/20/2020
'''

import pandas as pd
import statistics as st
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


def nestedCrossValidation(data, cVals, gammaVals, kFolds=10, mFolds=5):
    '''
    Nested 10x5 crossvalidation by default. Tests given cVals and gammaVals to determine
    best hyperparameters
    :param data: Data being trained on and tested on
    :param cVals: cVals being tested
    :param gammaVals: gammavals being tested
    :param kFolds: number of Outer folds
    :param mFolds: number of inner folds
    '''
    dataShuffled = data.sample(frac=1).reset_index(drop=True)

    colNames = getColNames(data)
    target = colNames[-1]

    numRows = len(dataShuffled)
    oneKthRows = round(numRows / kFolds)

    resultsTable = pd.DataFrame(columns=["OuterFold", "cVal", "gammaVal", "AUC", "TrainAccuracy", \
            "TrainPrecision", "TrainRecall", "TestAccuracy", "TestPrecision", "TestRecall"])

    _, ax = plt.subplots()

    for k in range(kFolds):

        testDataK = dataShuffled[(k * oneKthRows):oneKthRows * (k + 1)]
        trainDataK = dataShuffled.drop(dataShuffled.index[(k * oneKthRows):oneKthRows * (k + 1)])
        testDataK = testDataK.reset_index(drop=True)
        trainDataK = trainDataK.reset_index(drop=True)

        oneMthRows = round(len(trainDataK) / mFolds)

        bestAuc = 0
        bestC = 0
        bestGamma = 0

        for m in range(mFolds):

            testDataM = trainDataK[(m * oneMthRows):oneMthRows * (m + 1)]
            trainDataM = trainDataK.drop(trainDataK.index[(m * oneMthRows):oneMthRows * (m + 1)])

            # Try all combinations of gamma and C values from the lists given at each inner fold
            for g in gammaVals:
                for c in cVals:

                    x_train = trainDataM.drop(target, axis=1)
                    y_train = trainDataM[target]

                    x_test = testDataM.drop(target, axis=1)
                    y_test = testDataM[target]

                    # gaussian kernel
                    classifier = SVC(kernel='rbf', C=(c), gamma=g, probability=True)

                    # Train model
                    classifier.fit(x_train, y_train)

                    # TESTING model
                    y_pred = classifier.predict(x_test)
                    y_prob = classifier.predict_proba(x_test)
                    y_prob = y_prob[:, [1]]

                    acc = metrics.accuracy_score(y_test, y_pred)
                    prec = metrics.precision_score(y_test, y_pred)
                    rec = metrics.recall_score(y_test, y_pred)
                    rocAuc = metrics.roc_auc_score(y_test, y_prob)

                    print("OuterFold:", k, ", InnerFold:", m, ", C:", c, ", Gamma:", \
                           g, ", AUC:", rocAuc, ", Acc:", acc, ", Prec:", prec, ", Rec:", rec)

                    if rocAuc > bestAuc:
                        bestAuc = rocAuc
                        bestC = c
                        bestGamma = g

#         print(bestC, bestGamma, bestAcc)
        x_train = trainDataK.drop(target, axis=1)
        y_train = trainDataK[target]

        x_test = testDataK.drop(target, axis=1)
        y_test = testDataK[target]

        # gaussian kernel
        classifier = SVC(kernel='rbf', C=bestC, gamma=bestGamma, probability=True)

        # Train model
        classifier.fit(x_train, y_train)

        # TESTING model
        y_pred = classifier.predict(x_test)
        y_pred_train = classifier.predict(x_train)
        y_prob = classifier.predict_proba(x_test)
        y_prob = y_prob[:, [1]]

        acc = metrics.accuracy_score(y_test, y_pred)
        prec = metrics.precision_score(y_test, y_pred)
        rec = metrics.recall_score(y_test, y_pred)
        rocAuc = metrics.roc_auc_score(y_test, y_prob)

        accTrain = metrics.accuracy_score(y_train, y_pred_train)
        precTrain = metrics.precision_score(y_train, y_pred_train)
        recTrain = metrics.recall_score(y_train, y_pred_train)

        metrics.plot_roc_curve(classifier, x_test, y_test, name="AUC fold#" + str(k + 1), ax=ax)

        resultsTable.loc[len(resultsTable)] = [k + 1, bestC, bestGamma, rocAuc, accTrain, precTrain, recTrain, acc, prec, rec]

    cList = resultsTable["cVal"].tolist()
    gammaList = resultsTable["gammaVal"].tolist()
    aucList = resultsTable["AUC"].tolist()
    accListTr = resultsTable["TrainAccuracy"].tolist()
    precListTr = resultsTable["TrainPrecision"].tolist()
    recListTr = resultsTable["TrainRecall"].tolist()
    accList = resultsTable["TestAccuracy"].tolist()
    precList = resultsTable["TestPrecision"].tolist()
    recList = resultsTable["TestRecall"].tolist()

    resultsTable.loc[len(resultsTable)] = ["Avg", st.mean(cList), st.mean(gammaList)\
            , st.mean(aucList), st.mean(accListTr), st.mean(precListTr), st.mean(recListTr)\
            , st.mean(accList), st.mean(precList), st.mean(recList)]
    resultsTable.loc[len(resultsTable)] = ["StdDev", st.stdev(cList), st.stdev(gammaList)\
            , st.stdev(aucList), st.stdev(accListTr), st.stdev(precListTr), st.stdev(recListTr)\
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

    colNames = range(31)
    try:
        bCanData = pd.read_csv("datasets/breastcancer.csv", names=colNames)
    except:
        print("File not found")

    print(bCanData)

    cVals = [2 ** -5, 2 ** -3, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 8, 2 ** 10, 2 ** 12]
    gammaVals = [2 ** -15, 2 ** -12, 2 ** -9, 2 ** -3, 1, 2 ** 1, 2 ** 3, 2 ** 5]

    results = nestedCrossValidation(bCanData, cVals, gammaVals)

    pd.set_option('display.max_rows', None)

    print(results)
    plt.show()

'''



































'''

main()
