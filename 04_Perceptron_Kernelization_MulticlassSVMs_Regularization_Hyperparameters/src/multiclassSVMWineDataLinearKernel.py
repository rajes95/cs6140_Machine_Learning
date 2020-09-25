'''
CS 6140 Machine Learning - Assignment 04

Problem 4 - SVMs vs Multiclass Problems - Wine Dataset Linear Kernel

@author: Rajesh Sakhamuru
@version: 7/20/2020
'''

import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn import metrics

import perceptronFunctions as pf

import warnings
warnings.filterwarnings('ignore')


def nestedCrossValidation(data, cVals, kFolds=10, mFolds=5):
    '''
    Nested 10x5 crossvalidation by default. Tests given cVals to determine
    best hyperparameters
    :param data: Data being trained on and tested on
    :param cVals: cVals being tested
    :param kFolds: number of Outer folds
    :param mFolds: number of inner folds
    '''
    target = 0
    data = data.sample(frac=1).reset_index(drop=True)

    wineData1 = data.copy()
    wineData1.loc[wineData1[target] != 1, target] = 0
    wineData2 = data.copy()
    wineData2.loc[wineData2[target] != 2, target] = 0
    wineData3 = data.copy()
    wineData3.loc[wineData3[target] != 3, target] = 0

    numRows = len(data)
    oneKthRows = round(numRows / kFolds)

    resultsTable1 = pd.DataFrame(columns=["OuterFold", "cVal", "TrainAccuracy", \
            "TrainPrecision", "TrainRecall", "TestAccuracy", "TestPrecision", "TestRecall"])
    resultsTable2 = pd.DataFrame(columns=["OuterFold", "cVal", "TrainAccuracy", \
            "TrainPrecision", "TrainRecall", "TestAccuracy", "TestPrecision", "TestRecall"])
    resultsTable3 = pd.DataFrame(columns=["OuterFold", "cVal", "TrainAccuracy", \
            "TrainPrecision", "TrainRecall", "TestAccuracy", "TestPrecision", "TestRecall"])
    combinedResults = pd.DataFrame(columns=["OuterFold", "Combined TestAccuracy", \
                                            "Combined TestPrecision", "Combined TestRecall"])

    _, ax1 = plt.subplots(1)
    _, ax2 = plt.subplots(1)
    _, ax3 = plt.subplots(1)
    ax1.set_title("ROC-AUC Curve for Class 1")
    ax2.set_title("ROC-AUC Curve for Class 2")
    ax3.set_title("ROC-AUC Curve for Class 3")

    for k in range(kFolds):

        testDataK1 = wineData1[(k * oneKthRows):oneKthRows * (k + 1)]
        trainDataK1 = wineData1.drop(wineData1.index[(k * oneKthRows):oneKthRows * (k + 1)])
        testDataK1 = testDataK1.reset_index(drop=True)
        trainDataK1 = trainDataK1.reset_index(drop=True)

        testDataK2 = wineData2[(k * oneKthRows):oneKthRows * (k + 1)]
        trainDataK2 = wineData2.drop(wineData2.index[(k * oneKthRows):oneKthRows * (k + 1)])
        testDataK2 = testDataK2.reset_index(drop=True)
        trainDataK2 = trainDataK2.reset_index(drop=True)

        testDataK3 = wineData3[(k * oneKthRows):oneKthRows * (k + 1)]
        trainDataK3 = wineData3.drop(wineData3.index[(k * oneKthRows):oneKthRows * (k + 1)])
        testDataK3 = testDataK3.reset_index(drop=True)
        trainDataK3 = trainDataK3.reset_index(drop=True)

        oneMthRows = round(len(trainDataK1) / mFolds)

        bestAcc1 = 0
        bestC1 = 0

        bestAcc2 = 0
        bestC2 = 0

        bestAcc3 = 0
        bestC3 = 0

        for m in range(mFolds):

            testDataM1 = trainDataK1[(m * oneMthRows):oneMthRows * (m + 1)]
            trainDataM1 = trainDataK1.drop(trainDataK1.index[(m * oneMthRows):oneMthRows * (m + 1)])

            testDataM2 = trainDataK2[(m * oneMthRows):oneMthRows * (m + 1)]
            trainDataM2 = trainDataK2.drop(trainDataK2.index[(m * oneMthRows):oneMthRows * (m + 1)])

            testDataM3 = trainDataK3[(m * oneMthRows):oneMthRows * (m + 1)]
            trainDataM3 = trainDataK3.drop(trainDataK3.index[(m * oneMthRows):oneMthRows * (m + 1)])

            # normalize training and test data with zscore normalization
            trainDataM1, normParams, _ = pf.zScoreNormalization(trainDataM1, target)
            testDataM1, _, _ = pf.zScoreNormalization(testDataM1, target, normParams=normParams)
            trainDataM2, normParams, _ = pf.zScoreNormalization(trainDataM2, target)
            testDataM2, _, _ = pf.zScoreNormalization(testDataM2, target, normParams=normParams)
            trainDataM3, normParams, _ = pf.zScoreNormalization(trainDataM3, target)
            testDataM3, _, _ = pf.zScoreNormalization(testDataM3, target, normParams=normParams)

            # Try all combinations of gamma and C values from the lists given at each inner fold
            for c in cVals:

                x_train1 = trainDataM1.drop(target, axis=1)
                y_train1 = trainDataM1[target]
                x_test1 = testDataM1.drop(target, axis=1)
                y_test1 = testDataM1[target]

                x_train2 = trainDataM2.drop(target, axis=1)
                y_train2 = trainDataM2[target]
                x_test2 = testDataM2.drop(target, axis=1)
                y_test2 = testDataM2[target]

                x_train3 = trainDataM3.drop(target, axis=1)
                y_train3 = trainDataM3[target]
                x_test3 = testDataM3.drop(target, axis=1)
                y_test3 = testDataM3[target]

                # gaussian kernel
                classifier1 = SVC(kernel='linear', C=c, probability=True)
                classifier2 = SVC(kernel='linear', C=c, probability=True)
                classifier3 = SVC(kernel='linear', C=c, probability=True)

                # Train model
                classifier1.fit(x_train1, y_train1)
                classifier2.fit(x_train2, y_train2)
                classifier3.fit(x_train3, y_train3)

                # TESTING model
                y_pred1 = classifier1.predict(x_test1)
                y_pred2 = classifier2.predict(x_test2)
                y_pred3 = classifier3.predict(x_test3)

                y_prob1 = classifier1.predict_proba(x_test1)
                y_prob1 = y_prob1[:, [1]]

                y_prob2 = classifier2.predict_proba(x_test2)
                y_prob2 = y_prob2[:, [1]]

                y_prob3 = classifier3.predict_proba(x_test3)
                y_prob3 = y_prob3[:, [1]]

                acc1 = metrics.accuracy_score(y_test1, y_pred1)
                acc2 = metrics.accuracy_score(y_test2, y_pred2)
                acc3 = metrics.accuracy_score(y_test3, y_pred3)

                if acc1 > bestAcc1:
                    bestAcc1 = acc1
                    bestC1 = c
                if acc2 > bestAcc2:
                    bestAcc2 = acc2
                    bestC2 = c
                if acc3 > bestAcc3:
                    bestAcc3 = acc3
                    bestC3 = c

            print("Kfold:", k + 1, "Mfold:", m + 1, "Class1BestC:", bestC1, \
                   "Class2BestC:", bestC2, "Class3BestC:", bestC3)

        # normalize training and test data with zscore normalization
        trainDataK1, normParams, _ = pf.zScoreNormalization(trainDataK1, target)
        testDataK1, _, _ = pf.zScoreNormalization(testDataK1, target, normParams=normParams)
        trainDataK2, normParams, _ = pf.zScoreNormalization(trainDataK2, target)
        testDataK2, _, _ = pf.zScoreNormalization(testDataK2, target, normParams=normParams)
        trainDataK3, normParams, _ = pf.zScoreNormalization(trainDataK3, target)
        testDataK3, _, _ = pf.zScoreNormalization(testDataK3, target, normParams=normParams)

        x_train1 = trainDataK1.drop(target, axis=1)
        y_train1 = trainDataK1[target]
        x_test1 = testDataK1.drop(target, axis=1)
        y_test1 = testDataK1[target]

        x_train2 = trainDataK2.drop(target, axis=1)
        y_train2 = trainDataK2[target]
        x_test2 = testDataK2.drop(target, axis=1)
        y_test2 = testDataK2[target]

        x_train3 = trainDataK3.drop(target, axis=1)
        y_train3 = trainDataK3[target]
        x_test3 = testDataK3.drop(target, axis=1)
        y_test3 = testDataK3[target]

        # gaussian kernel
        classifier1 = SVC(kernel='linear', C=bestC1, probability=True)
        classifier2 = SVC(kernel='linear', C=bestC2, probability=True)
        classifier3 = SVC(kernel='linear', C=bestC3, probability=True)

        # Train model
        classifier1.fit(x_train1, y_train1)
        classifier2.fit(x_train2, y_train2)
        classifier3.fit(x_train3, y_train3)

        # TESTING model
        y_pred1 = classifier1.predict(x_test1)
        y_pred2 = classifier2.predict(x_test2)
        y_pred3 = classifier3.predict(x_test3)

        y_prob1 = classifier1.predict_proba(x_test1)
        y_prob1 = y_prob1[:, [1]]

        y_prob2 = classifier2.predict_proba(x_test2)
        y_prob2 = y_prob2[:, [1]]

        y_prob3 = classifier3.predict_proba(x_test3)
        y_prob3 = y_prob3[:, [1]]

        y_preds = multiclassPredict(y_prob1, y_prob2, y_prob3)
        y_tests = multiclassPredict(np.transpose([np.array(y_test1)]), np.transpose([np.array(y_test2)])\
                               , np.transpose([np.array(y_test3)]))

        acc1 = metrics.accuracy_score(y_test1, y_pred1)
        acc2 = metrics.accuracy_score(y_test2, y_pred2)
        acc3 = metrics.accuracy_score(y_test3, y_pred3)

        prec1 = metrics.precision_score(y_test1, y_pred1)
        prec2 = metrics.precision_score(y_test2, y_pred2, average=None)
        prec2 = prec2[1]
        prec3 = metrics.precision_score(y_test3, y_pred3, average=None)
        prec3 = prec3[1]

        rec1 = metrics.recall_score(y_test1, y_pred1)
        rec2 = metrics.recall_score(y_test2, y_pred2, average=None)
        rec2 = rec2[1]
        rec3 = metrics.recall_score(y_test3, y_pred3, average=None)
        rec3 = rec3[1]

        y_pred1Tr = classifier1.predict(x_train1)
        y_pred2Tr = classifier2.predict(x_train2)
        y_pred3Tr = classifier3.predict(x_train3)

        acc1Tr = metrics.accuracy_score(y_train1, y_pred1Tr)
        acc2Tr = metrics.accuracy_score(y_train2, y_pred2Tr)
        acc3Tr = metrics.accuracy_score(y_train3, y_pred3Tr)

        prec1Tr = metrics.precision_score(y_train1, y_pred1Tr)
        prec2Tr = metrics.precision_score(y_train2, y_pred2Tr, average=None)
        prec2Tr = prec2Tr[1]
        prec3Tr = metrics.precision_score(y_train3, y_pred3Tr, average=None)
        prec3Tr = prec3Tr[1]

        rec1Tr = metrics.recall_score(y_train1, y_pred1Tr)
        rec2Tr = metrics.recall_score(y_train2, y_pred2Tr, average=None)
        rec2Tr = rec2Tr[1]
        rec3Tr = metrics.recall_score(y_train3, y_pred3Tr, average=None)
        rec3Tr = rec3Tr[1]

        testAcc = metrics.accuracy_score(y_tests, y_preds)
        testPrec = metrics.precision_score(y_tests, y_preds, average='weighted')
        testRec = metrics.recall_score(y_tests, y_preds, average='weighted')

        resultsTable1.loc[len(resultsTable1)] = [k + 1, bestC1, acc1Tr, prec1Tr, rec1Tr, acc1, prec1, rec1]
        resultsTable2.loc[len(resultsTable2)] = [k + 1, bestC2, acc2Tr, prec2Tr, rec2Tr, acc2, prec2, rec2]
        resultsTable3.loc[len(resultsTable3)] = [k + 1, bestC3, acc3Tr, prec3Tr, rec3Tr, acc3, prec3, rec3]
        combinedResults.loc[len(combinedResults)] = [k + 1, testAcc, testPrec, testRec]

        metrics.plot_roc_curve(classifier1, x_test1, y_test1, name="AUC fold#" + str(k + 1), ax=ax1)
        metrics.plot_roc_curve(classifier2, x_test2, y_test2, name="AUC fold#" + str(k + 1), ax=ax2)
        metrics.plot_roc_curve(classifier3, x_test3, y_test3, name="AUC fold#" + str(k + 1), ax=ax3)

    cList = resultsTable1["cVal"].tolist()
    accListTr = resultsTable1["TrainAccuracy"].tolist()
    precListTr = resultsTable1["TrainPrecision"].tolist()
    recListTr = resultsTable1["TrainRecall"].tolist()
    accList = resultsTable1["TestAccuracy"].tolist()
    precList = resultsTable1["TestPrecision"].tolist()
    recList = resultsTable1["TestRecall"].tolist()
    resultsTable1.loc[len(resultsTable1)] = ["Avg", st.mean(cList), \
        st.mean(accListTr), st.mean(precListTr), st.mean(recListTr), st.mean(accList), st.mean(precList), st.mean(recList)]
    resultsTable1.loc[len(resultsTable1)] = ["StdDev", st.stdev(cList), \
        st.stdev(accListTr), st.stdev(precListTr), st.stdev(recListTr), st.stdev(accList), st.stdev(precList), st.stdev(recList)]

    cList = resultsTable2["cVal"].tolist()
    accListTr = resultsTable2["TrainAccuracy"].tolist()
    precListTr = resultsTable2["TrainPrecision"].tolist()
    recListTr = resultsTable2["TrainRecall"].tolist()
    accList = resultsTable2["TestAccuracy"].tolist()
    precList = resultsTable2["TestPrecision"].tolist()
    recList = resultsTable2["TestRecall"].tolist()
    resultsTable2.loc[len(resultsTable2)] = ["Avg", st.mean(cList), \
        st.mean(accListTr), st.mean(precListTr), st.mean(recListTr), st.mean(accList), st.mean(precList), st.mean(recList)]
    resultsTable2.loc[len(resultsTable2)] = ["StdDev", st.stdev(cList), \
        st.stdev(accListTr), st.stdev(precListTr), st.stdev(recListTr), st.stdev(accList), st.stdev(precList), st.stdev(recList)]

    cList = resultsTable3["cVal"].tolist()
    accListTr = resultsTable3["TrainAccuracy"].tolist()
    precListTr = resultsTable3["TrainPrecision"].tolist()
    recListTr = resultsTable3["TrainRecall"].tolist()
    accList = resultsTable3["TestAccuracy"].tolist()
    precList = resultsTable3["TestPrecision"].tolist()
    recList = resultsTable3["TestRecall"].tolist()
    resultsTable3.loc[len(resultsTable3)] = ["Avg", st.mean(cList), \
        st.mean(accListTr), st.mean(precListTr), st.mean(recListTr), st.mean(accList), st.mean(precList), st.mean(recList)]
    resultsTable3.loc[len(resultsTable3)] = ["StdDev", st.stdev(cList), \
        st.stdev(accListTr), st.stdev(precListTr), st.stdev(recListTr), st.stdev(accList), st.stdev(precList), st.stdev(recList)]

    accList = combinedResults["Combined TestAccuracy"].tolist()
    precList = combinedResults["Combined TestPrecision"].tolist()
    recList = combinedResults["Combined TestRecall"].tolist()
    combinedResults.loc[len(combinedResults)] = ["Avg", st.mean(accList), st.mean(precList), st.mean(recList)]
    combinedResults.loc[len(combinedResults)] = ["StdDev", st.stdev(accList), st.stdev(precList), st.stdev(recList)]

    return resultsTable1, resultsTable2, resultsTable3, combinedResults


def getColNames(data):
    '''
    Returns list of column names of data given
    :param data: pandas dataframe
    '''
    return list(data.columns)


def multiclassPredict(y_prob1, y_prob2, y_prob3):
    '''
    Uses Predicted probabilities for all 3 classes to make best predictions for 
    that particular datapoint
    :param y_prob1: class 1 probabilities
    :param y_prob2: class 2 probabilities
    :param y_prob3: class 3 probabilities
    '''
    df = pd.DataFrame()
    df[1] = y_prob1.T.tolist()[0]
    df[2] = y_prob2.T.tolist()[0]
    df[3] = y_prob3.T.tolist()[0]
    df["a"] = range(len(df))

    a = 0
    for n in range(len(df)):
        if df[1][n] >= df[2][n] and df[1][n] > df[3][n]:
            a = 1
        elif df[2][n] > df[1][n] and df[2][n] >= df[3][n]:
            a = 2
        elif df[3][n] >= df[1][n] and df[3][n] > df[2][n]:
            a = 3
        df["a"][n] = a

#     print(df)
#     print(np.array(df["a"]))
    return np.transpose([np.array(df["a"])])


def main():
    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    # import data file

    colNames = range(14)
    try:
        wineData = pd.read_csv("datasets/wine.csv", names=colNames)
    except:
        print("File not found")

    print(wineData)
    cVals = [2 ** -5, 2 ** -3, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 8, 2 ** 10]

    resultsTable1, resultsTable2, resultsTable3, combinedResults \
 = nestedCrossValidation(wineData, cVals)

    print("\nClass 1 Results:")
    print(resultsTable1)
    print("\nClass 2 Results:")
    print(resultsTable2)
    print("\nClass 3 Results:")
    print(resultsTable3)
    print("\nOverall Results (Predictions made using all 3 models concurrently):")
    print(combinedResults)

    plt.show()

'''



































'''

main()
