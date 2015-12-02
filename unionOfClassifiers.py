from nltk.classify.decisiontree import DecisionTreeClassifier
from numba.cuda.cudadrv.devices import _Runtime


def runMethod2():
    __author__ = 'Umberto'
    import pandas
    import numpy
    from sklearn import metrics,svm,linear_model
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
    from sklearn.cross_validation import cross_val_score
    import os
    from os.path import join, isfile
    from collections import defaultdict
    from sklearn.naive_bayes import GaussianNB,BernoulliNB
    import pandas as pn
    from errorAnalysis import meanError
    from sklearn.metrics import confusion_matrix

    averageError = 0
    listUsers = list()
    myPath = 'trainPerRank/'

    listUsers = [f for f in os.listdir(str(myPath)) if isfile(join(myPath, f))]
    #listUsers.remove('.DS_Store')

    dictResults = defaultdict(float)
    confusionMatrix= numpy.array([[0, 0, 0,0,0],
            [0, 0, 0,0,0],
           [0, 0, 0,0,0],
                 [0, 0, 0,0,0],
                 [0, 0, 0,0,0]])

    listOfPredictions = []
    listOfTrainData=[]
    listOfTrainTarget = []
    listOfTestData = []
    listOfTestTarget = []


    for user in listUsers:
         #======READING TRAIN SET========
        dataTrain_12 = numpy.array(pandas.read_csv('trainPerRank/1-2/prediction/'+user, header=None))
        trainRank = list(map(lambda x:x[1],numpy.array(pandas.read_csv('trainPerRank/1-2/stars/'+user, header=None))))
        #Need flat list
        dataTrain_12 = [val for sublist in dataTrain_12 for val in sublist]


        dataTrain_3 = numpy.array(pandas.read_csv('trainPerRank/3/prediction/'+user, header=None))
        #Need flat list
        dataTrain_3 = [val for sublist in dataTrain_3 for val in sublist]

        dataTrain_45 = numpy.array(pandas.read_csv('trainPerRank/4-5/prediction/'+user, header=None))
         #Need flat list
        dataTrain_45 = [val for sublist in dataTrain_45 for val in sublist]

        dfData = pn.DataFrame(pn.Series(dataTrain_12))

        dfData = pn.concat([dfData, pn.Series(dataTrain_3)], axis=1)
        dfData = pn.concat([dfData, pn.Series(dataTrain_45)], axis=1)

        dataTrain = numpy.array(dfData)

        #===========SAVE ENSEMBLED DATATRAIN (for neural))================
        listOfTrainData.append(dataTrain)
        listOfTrainTarget.append(trainRank)


        #============READING TEST SET ==========
        dataTest_12 = numpy.array(pandas.read_csv('testPerRank/1-2/prediction/'+user, header=None))
        dataTest_3 = numpy.array(pandas.read_csv('testPerRank/3/prediction/'+user, header=None))
        dataTest_45 = numpy.array(pandas.read_csv('testPerRank/4-5/prediction/'+user, header=None))

        dataTest_12 = [val for sublist in dataTest_12 for val in sublist]
        dataTest_3 = [val for sublist in dataTest_3 for val in sublist]
        dataTest_45 = [val for sublist in dataTest_45 for val in sublist]

        dfDataTest = pn.DataFrame(pn.Series(dataTest_12))

        dfDataTest = pn.concat([dfDataTest, pn.Series(dataTest_3)], axis=1)
        dfDataTest = pn.concat([dfDataTest, pn.Series(dataTest_45)], axis=1)

        dataTest = numpy.array(dfDataTest)

        testRank = list(map(lambda x:x[1],numpy.array(pandas.read_csv('testPerRank/1-2/stars/'+user, header=None))))

        #=========SAVE TESTSET (neural)=========
        listOfTestData.append(dataTest)
        listOfTestTarget.append(testRank)

        # gnb = GaussianNB()
        # gnb.fit(dataTrain, trainRank)
        # prediction = gnb.predict(dataTest)
        #
        # treeClassifier = tree.DecisionTreeClassifier()
        # treeClassifier.fit(dataTrain, trainRank)
        # prediction = treeClassifier.predict(dataTest)
#         #
#          0.534526167607
# =============ERROR ANALYSIS=========
# -0.0107943569909
# confusion matrix:
# [[  0   9   8   1   4]
#  [  1  32  19  30  10]
#  [  0  33 143  95  30]
#  [  0  38  67 400  72]
#  [  0  20  20 138 147]]

        # clf = svm.SVC(kernel='linear')
        # clf.decision_function_shape = 'ovr'
        # clf.fit(dataTrain, trainRank)
        # prediction = clf.predict(dataTest)
# =============Accuracy=========
# 0.53582203379
# =============ERROR ANALYSIS=========
# -0.0183031215174
# confusion matrix:
# [[  0   9   8   1   4]
#  [  1  32  19  30  10]
#  [  1  32 137 100  31]
#  [  0  38  59 406  74]
#  [  0  20  19 138 148]]
          #MAXENT CREATION
        # logreg = linear_model.LogisticRegression()
        # logreg.solver = 'lbfgs'
        # logreg.fit(dataTrain, trainRank)
        # #==========MAXENT PREDICTION ========
        # prediction = logreg.predict(dataTest)#0.5349

        # forest = RandomForestClassifier(n_estimators=40)
        # forest.fit(dataTrain, trainRank)
        # prediction = forest.predict(dataTest)
        clf1 = DecisionTreeClassifier(max_depth=1)
        clf = AdaBoostClassifier(base_estimator=clf1,
                                 algorithm="SAMME.R",
                                 n_estimators=3)
        clf.fit(dataTrain, trainRank)
        prediction = clf.predict(dataTest)
#          =============Accuracy=========
# 0.538671496693
# =============ERROR ANALYSIS=========
# -0.075507041171
# confusion matrix:
# [[  0   5   8   5   4]
#  [  1  17  20  43  11]
#  [  0  17 144 113  27]
#  [  0  22  67 422  66]
#  [  0  14  18 153 140]]


        dictResults[user] = metrics.accuracy_score(testRank, prediction)

        averageError += meanError(prediction,testRank)

        confusionMatrix  += confusion_matrix(testRank,prediction,labels=[1,2,3,4,5])

        listOfPredictions.append(prediction)

    accuracy = float()
    for user in dictResults.keys():
        accuracy += dictResults[user]

    accuracy /= len(listUsers)
    print('=============Accuracy=========')

    print(accuracy)
    #with balanced train 0.5063370
    #with unbalanced train 0.515719385728

    print('=============ERROR ANALYSIS=========')
    print(averageError/len(listUsers)) #=> 0.00794524594896

    print('confusion matrix:')
    print(confusionMatrix)
    # confusion matrix:
    # [[  4   5   8   1   4]
    #  [ 16  17  19  25  15]
    #  [ 13  26 139  88  35]
    #  [ 23  34  60 360 100]
    #  [  6  30  22 112 155]]

    listOfTrainData = [val for sublist in listOfTrainData for val in sublist]
    listOfTestData = [val for sublist in listOfTestData for val in sublist]
    listOfTrainTarget = [val for sublist in listOfTrainTarget for val in sublist]
    listOfTestTarget = [val for sublist in listOfTestTarget for val in sublist]

    dfTrain = pn.DataFrame(listOfTrainData)
    dfTrainRank =pn.DataFrame()
    for val in listOfTrainTarget:
        if val==1:
            dfTrainRank=dfTrainRank.append(pn.Series([0,0,0,0,1]),ignore_index=True)
        elif val ==2:
            dfTrainRank=dfTrainRank.append(pn.Series([0,0,0,1,0]),ignore_index=True)
        elif val ==3:
            dfTrainRank=dfTrainRank.append(pn.Series([0,0,1,0,0]),ignore_index=True)
        elif val ==4:
            dfTrainRank=dfTrainRank.append(pn.Series([0,1,0,0,0]),ignore_index=True)
        elif val ==5:
            dfTrainRank=dfTrainRank.append(pn.Series([1,0,0,0,0]),ignore_index=True)

    #create folder if necessary
    if not os.path.exists('trainForNeural/'):
        os.makedirs('trainForNeural/')

    dfTrain.to_csv('trainForNeural/data.csv', header=False, index_label=False, index=False)
    dfTrainRank.to_csv('trainForNeural/target.csv', header=False, index_label=False, index=False)


    dfTest = pn.DataFrame(listOfTestData)
    dfTestRank =pn.DataFrame()
    for val in listOfTestTarget:
        if val==1:
            dfTestRank=dfTestRank.append(pn.Series([0,0,0,0,1]),ignore_index=True)
        elif val ==2:
            dfTestRank=dfTestRank.append(pn.Series([0,0,0,1,0]),ignore_index=True)
        elif val ==3:
            dfTestRank=dfTestRank.append(pn.Series([0,0,1,0,0]),ignore_index=True)
        elif val ==4:
            dfTestRank=dfTestRank.append(pn.Series([0,1,0,0,0]),ignore_index=True)
        elif val ==5:
            dfTestRank=dfTestRank.append(pn.Series([1,0,0,0,0]),ignore_index=True)

    #create folder if necessary
    if not os.path.exists('testForNeural/'):
        os.makedirs('testForNeural/')

    dfTest.to_csv('testForNeural/data.csv', header=False, index_label=False, index=False)
    dfTestRank.to_csv('testForNeural/target.csv', header=False, index_label=False, index=False)

    return listOfPredictions

runMethod2()