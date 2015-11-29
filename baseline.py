import os
from os.path import join, isfile
import pandas
import numpy
from collections import defaultdict
from sklearn import metrics
from errorAnalysis import *


myPath = 'trainAll/'
listUsers = [f for f in os.listdir(str(myPath)) if isfile(join(myPath, f))]
#listUsers.remove('.DS_Store')

dictResult = defaultdict(float)

averageError = float()
errorSet = int()
for user in listUsers:
    dictCount = defaultdict(int)
    majority = float()
    trainRank = numpy.array(pandas.read_csv('trainAll/stars/'+user, header=None))
    trainRank = [val for sublist in trainRank for val in sublist]
    trainRank = list(map(lambda x: int(x*5), trainRank))

    validationRank = numpy.array(pandas.read_csv('validationAll/stars/'+user, header=None))
    validationRank = [val for sublist in validationRank for val in sublist]
    validationRank = list(map(lambda x: int(x*5), validationRank))

    testRank = numpy.array(pandas.read_csv('testAll/stars/'+user, header=None))
    testRank = [val for sublist in testRank for val in sublist]
    testRank = list(map(lambda x: int(x*5), testRank))

    for x in trainRank:
        dictCount[x] += 1
    for x in validationRank:
        dictCount[x] += 1

    for x in dictCount.keys():
        if(dictCount[x] == max(dictCount.values())):
            majority = x
            break

    prediction = list()
    for x in range(len(testRank)):
        prediction.append(majority)

    dictResult[user] = metrics.accuracy_score(testRank, prediction)

    averageError += meanError(prediction,testRank)
    errorSet += setError(prediction,testRank)

accuracy = float()
for key in dictResult.keys():
    accuracy += dictResult[key]

accuracy /= len(listUsers)

print(accuracy)
print('=============ERROR=========')
print(averageError/len(listUsers)) #=> -0.15346161033753716
print(errorSet) #=> 736


# trainAll, minReview 20: 0.441929922377 -> test same
# trainAll, minReview 50: 0.449429999815 -> test 0.454907309474




