__author__ = 'vittorioselo'

import pandas
import numpy
from sklearn import metrics
import os
from os.path import join, isfile
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor



listUsers = list()
myPath = 'train/'

listUsers = [f for f in os.listdir(str(myPath)) if isfile(join(myPath, f))]
listUsers.remove('.DS_Store')

dictResults = defaultdict(float)

i = 0
for user in listUsers:
    i += 1
    print(i)
    #======READING TRAIN SET========
    dataTrain = numpy.array(pandas.read_csv('train/'+user, header=None))
    trainRank = numpy.array(pandas.read_csv('train/stars/'+user, header=None))

    #Need flat list
    trainRank = [val for sublist in trainRank for val in sublist]
    trainRank = list(map(lambda x: int(x*5), trainRank))

    #============READING TEST SET ==========
    dataTest = numpy.array(pandas.read_csv('test/'+user, header=None))
    testRank = numpy.array(pandas.read_csv('test/stars/'+user, header=None))

    testRank = [val for sublist in testRank for val in sublist]
    testRank = list(map(lambda x: int(x*5), testRank))

    forest = RandomForestClassifier(n_estimators = 200)
    forest.fit(dataTrain, trainRank)

    prediction = forest.predict(dataTest)

    dictResults[user] = metrics.accuracy_score(testRank, prediction)

accuracy = float()
for user in dictResults.keys():
    accuracy += dictResults[user]

accuracy /= len(listUsers)

print(accuracy)

