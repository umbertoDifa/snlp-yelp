__author__ = 'Umberto'
import pandas
import numpy
from sklearn import metrics
import os
from os.path import join, isfile
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
import pandas as pn
from errorAnalysis import meanError

averageError = 0
listUsers = list()
myPath = 'trainPerRank/'

listUsers = [f for f in os.listdir(str(myPath)) if isfile(join(myPath, f))]
#listUsers.remove('.DS_Store')

dictResults = defaultdict(float)

for user in listUsers:
     #======READING TRAIN SET========
    dataTrain_12 = numpy.array(pandas.read_csv('trainPerRank/1-2/prediction/'+user, header=None))
    trainRank = list(map(lambda x:x[1],numpy.array(pandas.read_csv('trainPerRank/1-2/stars/'+user, header=None))))
    #Need flat list
    #trainRank = [val for sublist in trainRank for val in sublist]
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

    gnb = GaussianNB()
    gnb.fit(dataTrain, trainRank)
    prediction = gnb.predict(dataTest)

    dictResults[user] = metrics.accuracy_score(testRank, prediction)

    averageError += meanError(prediction,testRank)


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


