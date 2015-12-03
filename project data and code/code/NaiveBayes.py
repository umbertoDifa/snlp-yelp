__author__ = 'vittorioselo'
import pandas
import numpy
from sklearn import metrics
import os
from os.path import join, isfile
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB



listUsers = list()
myPath = 'train/'

listUsers = [f for f in os.listdir(str(myPath)) if isfile(join(myPath, f))]
listUsers.remove('.DS_Store')

dictResults = defaultdict(float)

for user in listUsers:
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

    gnb = GaussianNB()
    gnb.fit(dataTrain, trainRank)
    prediction = gnb.predict(dataTest)

    dictResults[user] = metrics.accuracy_score(testRank, prediction)

accuracy = float()
for user in dictResults.keys():
    accuracy += dictResults[user]

accuracy /= len(listUsers)

print(accuracy)
#MIN REVIEWS 20
# ACC .389963452794 -> NO TAG
# ACC .332423606817 -> noun
# ACC .369191294046 -> noun + adjective

#MIN REVIEWS 25
#ACC .339427339762 -> noun


