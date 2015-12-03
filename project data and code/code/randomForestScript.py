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

    forest = RandomForestClassifier(n_estimators=400)
    forest.fit(dataTrain, trainRank)

    prediction = forest.predict(dataTest)

    dictResults[user] = metrics.accuracy_score(testRank, prediction)

accuracy = float()
for user in dictResults.keys():
    accuracy += dictResults[user]

accuracy /= len(listUsers)

print(accuracy)
#MIN REVIEWS 20
#ACC .416982853394 -> NO TAGS 400 tree
#ACC .415015311991 -> noun and 100 200 tree
#ACC .415776112382 -> noun + adj and 200 trees
#ACC .445425589304 -> noun + adj and 400 trees
#ACC .44691447588 -> noun + adj and 1000 trees


#MIN REVIEWS 25 ONLY NOUND
#ACC .425177274014