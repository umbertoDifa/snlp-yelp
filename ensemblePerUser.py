__author__ = 'vittorioselo'


import pandas
import numpy
from sklearn import svm
from sklearn import metrics
import os
from os.path import join, isfile
from collections import defaultdict
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier


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

     #=========READING VALIDATION SET =========
    dataValidation = numpy.array(pandas.read_csv('validation/'+user, header=None))
    validationRank = numpy.array(pandas.read_csv('validation/stars/'+user, header=None))


    validationRank = [val for sublist in validationRank for val in sublist]
    validationRank = list(map(lambda x: int(x*5), validationRank))

    #============READING TEST SET ==========
    dataTest = numpy.array(pandas.read_csv('test/'+user, header=None))
    testRank = numpy.array(pandas.read_csv('test/stars/'+user, header=None))

    testRank = [val for sublist in testRank for val in sublist]
    testRank = list(map(lambda x: int(x*5), testRank))

    #======SVM=========
    clf1 = svm.SVC()#RBF
    clf1.decision_function_shape = 'ovr'
    clf1.fit(dataTrain, trainRank)

    clf2 = svm.SVC() #LINEAR
    clf2.decision_function_shape ='ovr'
    clf2.kernel = 'linear'
    clf2.fit(dataTrain, trainRank)
    #==========MAX ENT ==========
    logreg = linear_model.LogisticRegression()
    logreg.solver = 'lbfgs'
    logreg.fit(dataTrain, trainRank)

    #========RANDOM FOREST ==========
    #forest = RandomForestClassifier(n_estimators=200)
    #forest.fit(dataTrain, trainRank)

    #========CHOOSING PREDICTOR BASE ON VALDATION SET=========
    pre1 = clf1.predict(dataValidation)
    pre2 = clf2.predict(dataValidation)
    pre3 = logreg.predict(dataValidation)
    #pre4 = forest.predict(dataValidation)

    acc1 = metrics.accuracy_score(validationRank, pre1)
    acc2 = metrics.accuracy_score(validationRank, pre2)
    acc3 = metrics.accuracy_score(validationRank, pre3)
    #acc4 = metrics.accuracy_score(validationRank, pre4)
    #print('============')
    #print(user)
    #print(acc1)
    #print(acc2)
    #print(acc3)
    #print(acc4)

    prediction = float()
    #if(acc4 >= acc1 and acc4 >= acc2 and acc4 >= acc3):
        #prediction = forest.predict(dataTest)
        #print('4')
    if(acc1 >= acc2 and acc1>= acc3):
        prediction = clf1.predict(dataTest)
        #print('1')
    elif(acc2 >= acc3):
        prediction = clf2.predict(dataTest)
        #print('2')
    else:
        prediction = logreg.predict(dataTest)
        #print('3')

    dictResults[user] = metrics.accuracy_score(testRank, prediction)

accuracy = float()

for user in dictResults.keys():
        accuracy += dictResults[user]
    #print('========')
    #print(user)
    #print(dictResults[user])

accuracy /= len(listUsers)

print(accuracy) #ACC 0.457867937661

#BASELINE
#ACC 0.449273749986