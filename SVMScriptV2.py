__author__ = 'vittorioselo'

def runTest(rank):
    import pandas
    import numpy
    from sklearn import svm
    from sklearn import metrics
    import os
    from os.path import join, isfile
    from collections import defaultdict
    from sklearn import linear_model
    import pandas as pn
    from errorAnalysis import meanError


    listUsers = list()
    myPath = 'trainPerRank/'

    listUsers = [f for f in os.listdir(str(myPath)) if isfile(join(myPath, f))]
    #listUsers.remove('.DS_Store')

    dictResult = defaultdict(list)

    dictResultMaxEnt = defaultdict(list)

    dictPredictor = defaultdict(list)

    dictRealResult = defaultdict(float)

    dictSVMTest = defaultdict(float)
    dictMAXENTTest = defaultdict(float)
    averageError = 0

    for user in listUsers:
        #======READING TRAIN SET========
        dataTrain = numpy.array(pandas.read_csv('trainPerRank/'+user, header=None))
        #take the real scores
        trainRank = list(map(lambda x:x[0],numpy.array(pandas.read_csv('trainPerRank/'+rank+'/stars/'+user, header=None))))

        #if rank '1-2' balance the dataset
        # if rank == '1-2':
        #     tot = len(trainRank) #how many datapoint
        #     other=[i for i, j in enumerate(trainRank) if j == -1] #how many 'other'
        #     otherToKeep = other[:tot-len(other)] #save index of 'other' to keep
        #     correctToKeep = [i for i, j in enumerate(trainRank) if j != -1]
        #     keepers = otherToKeep+correctToKeep
        #
        #     #slice the rank file
        #     trainRank=[j for i,j in enumerate(trainRank) if i in keepers]
        #     #slice the datafile
        #     dataTrain=[j for i,j in enumerate(dataTrain) if i in keepers]


        #=========READING VALIDATION SET =========
        dataValidation = numpy.array(pandas.read_csv('validationPerRank/'+user, header=None))
        testValidation = list(map(lambda x:x[0],numpy.array(pandas.read_csv('validationPerRank/'+rank+'/stars/'+user, header=None))))


        #SVM CREATION
        clf = svm.SVC()
        #MAXENT CREATION
        logreg = linear_model.LogisticRegression()
        logreg.solver = 'lbfgs'
        #========SVM FITTING ========
        #======== ONE VS ALL ===========
        #clf.kernel = 'sigmoid' #ACC .45530770854 / WEIGHTACC .449320486479
        #clf.kernel = 'linear' #ACC .447685809102 / WEIGHTACC .440899375124
        #clf.kernel = 'poly' #ACC .457268492854 / WEIGHTACC .451035542219 /DEGREE 3
        #clf.degree = 1   #ACC .45099398305 / WEIGHTACC .447188917203
        #clf.kernel = 'rbf' #ACC .45099398305 / WEIGHTACC .447188917203
        clf.decision_function_shape = 'ovr'
        clf.fit(dataTrain, trainRank)
        #===========SVM PREDICTION ==========
        predicted = clf.predict(dataValidation)

        #===========MAXENT FITTING ==========
        logreg.fit(dataTrain, trainRank) #ACC .44
        #==========MAXENT PREDICTION ========
        maxentPrediction = logreg.predict(dataValidation)
        #SAVING RESULT SVM
        dictResult[user].append(len(dataTrain)+len(dataValidation))
        dictResult[user].append(metrics.accuracy_score(testValidation, predicted))
        #SAVING RESULT MAXENT
        dictResultMaxEnt[user].append(len(dataTrain)+len(dataValidation))
        dictResultMaxEnt[user].append(metrics.accuracy_score(testValidation, maxentPrediction))
        #SAVING THE TWO PREDICTOR PER USER
        dictPredictor[user].append(clf)
        dictPredictor[user].append(logreg)

        #============PREDICTION ON TEST==============
        dataTest = numpy.array(pandas.read_csv('testPerRank/'+user, header=None))
        testRank = list(map(lambda x:x[0],numpy.array(pandas.read_csv('testPerRank/'+rank+'/stars/'+user, header=None))))

        #testRank = [val for sublist in testRank for val in sublist]
        #testRank = list(map(lambda x: int(x+1), testRank))

        pre1 = clf.predict(dataTest)
        pre2 = logreg.predict(dataTest)
        dictSVMTest[user] = metrics.accuracy_score(testRank, pre1)
        dictMAXENTTest[user] = metrics.accuracy_score(testRank, pre2)

    #ACCURACY CALCULATION for SVM
    accuracy = float()
    weightedAccuracy = float()
    tot = float()

    #======ACCURACY CALCULATION FOR MAX ENT
    accuracyMaxEnt = float()
    weightedAccuracyMaxEnt = float()


    for value in dictResult.keys():
        accuracy += dictResult[value][1]
        weightedAccuracy += (dictResult[value][1]*dictResult[value][0])
        accuracyMaxEnt += dictResultMaxEnt[value][1]
        weightedAccuracyMaxEnt += (dictResultMaxEnt[value][1]*dictResultMaxEnt[value][0])
        tot += dictResult[value][0]

    accuracy /= len(listUsers)
    weightedAccuracy /= tot

    accuracyMaxEnt /= len(listUsers)
    weightedAccuracyMaxEnt /= tot

    #SVM RESULT
    print('SVM-VALIDATION')
    print(accuracy)
    print(weightedAccuracy)
    print('MAXENT-VALIDATION')
    print(accuracyMaxEnt)
    print(weightedAccuracyMaxEnt)


    #===== CALCULATE ACCURACY PER RANGE ======
    range = 20
    for x in [20,40,60,80,100]:
        users = list()
        accuracy = float()
        weightedAccuracy =float ()
        tot = int()
        counter = 0
        accuracyMaxEnt = float()
        weightedAccuracyMaxEnt = float()
        for value in dictResult.keys():
            if(dictResult[value][0]<= x and dictResult[value][0] > (x-range)):
                users.append(value)
                accuracy += dictResult[value][1]
                weightedAccuracy += (dictResult[value][1]*dictResult[value][0])
                tot += dictResult[value][0]
                counter+=1
                accuracyMaxEnt += dictResultMaxEnt[value][1]
                weightedAccuracyMaxEnt += (dictResultMaxEnt[value][1]*dictResultMaxEnt[value][0])
        if(not counter==0):
            accuracy /= counter
            accuracyMaxEnt /= counter
        else:
            accuracy=0
            accuracyMaxEnt = 0
        if(not tot == 0):
            weightedAccuracy /= tot
            weightedAccuracyMaxEnt /= tot
        else:
            weightedAccuracy = 0
            weightedAccuracyMaxEnt =0
        for x in users:
            if(accuracy > accuracyMaxEnt):
                del dictPredictor[x][1]
            else:
                del dictPredictor[x][0]

        #print('SVM')
        #print('('+str(tot)+')Accuracy for users with review between '+str(x-range)+' and '+str(x)+' is: '+str(accuracy))
        #print('('+str(tot)+')Weighted Accuracy for users with review between '+str(x-range)+' and '+str(x)+' is: '+str(weightedAccuracy))
        #print('MAXENT')
        #print('('+str(tot)+')Accuracy for users with review between '+str(x-range)+' and '+str(x)+' is: '+str(accuracyMaxEnt))
        #print('('+str(tot)+')Weighted Accuracy for users with review between '+str(x-range)+' and '+str(x)+' is: '+str(weightedAccuracyMaxEnt))

    #=====ANALYSING THE REMAINING ONE >= 100
    accuracy = float()
    weightedAccuracy = float()
    tot = int()
    counter = 0
    accuracyMaxEnt = float()
    weightedAccuracyMaxEnt = float()
    users = list()
    for value in dictResult.keys():
        if(dictResult[value][0]> 100):
            users.append(value)
            accuracy += dictResult[value][1]
            weightedAccuracy += (dictResult[value][1]*dictResult[value][0])
            tot += dictResult[value][0]
            counter+=1
            accuracyMaxEnt += dictResultMaxEnt[value][1]
            weightedAccuracyMaxEnt += (dictResultMaxEnt[value][1]*dictResultMaxEnt[value][0])
    if(not counter == 0):
        accuracy /= counter
        accuracyMaxEnt /= counter
    else:
        accuracy=0
        accuracyMaxEnt
    if(not tot == 0):
        weightedAccuracy /= tot
        weightedAccuracyMaxEnt /= tot
    else:
        weightedAccuracy = 0
        weightedAccuracyMaxEnt = 0
    for x in users:
        if(accuracy > accuracyMaxEnt):
            del dictPredictor[x][1]
        else:
            del dictPredictor[x][0]
    #print('SVM')
    #print('('+str(tot)+')Accuracy for users with review greater than '+str(x)+' is: '+str(accuracy))
    #print('('+str(tot)+')Weighted Accuracy for users with review greater than '+str(x)+' is: '+str(weightedAccuracy))
    #print('MAXENT')
    #print('('+str(tot)+')Accuracy for users with review greater than '+str(x)+' is: '+str(accuracyMaxEnt))
    #print('('+str(tot)+')Weighted Accuracy for users with review greater than '+str(x)+' is: '+str(weightedAccuracyMaxEnt))

    #========PREDICTION ON TRAIN TO CREATE DATA FOR THE SECOND CLASSIFIER====#
    for user in dictPredictor.keys():
        dataTest = numpy.array(pandas.read_csv('trainPerRank/'+user, header=None))

        predicted = dictPredictor[user][0].predict(dataTest)
        dfPrediction = pn.DataFrame(pn.Series(predicted))

        dfPrediction.to_csv('trainPerRank/'+rank+'/prediction/'+user, header=False, index_label=False, index=False)

    #=========== PREDICTION ON REAL DATASET =========#

    for user in dictPredictor.keys():
        dataTest = numpy.array(pandas.read_csv('testPerRank/'+user, header=None))
        testRank = list(map(lambda x:x[0],numpy.array(pandas.read_csv('testPerRank/'+rank+'/stars/'+user, header=None))))

        predicted = dictPredictor[user][0].predict(dataTest)
        dfPrediction = pn.DataFrame(pn.Series(predicted))
        dfPrediction.to_csv('testPerRank/'+rank+'/prediction/'+user, header=False, index_label=False, index=False)

        dictRealResult[user] = metrics.accuracy_score(testRank, predicted)


    accuracy = float()
    for key in dictMAXENTTest.keys():
        accuracy += dictMAXENTTest[key]
    print('MAXENT TEST')
    print(accuracy/len(listUsers))
    accuracy = float()
    for key in dictSVMTest.keys():
        accuracy += dictSVMTest[key]
    print('SVM TEST')
    print(accuracy/len(listUsers))
    accuracy = float()
    for key in dictRealResult.keys():
        accuracy += dictRealResult[key]

    accuracy /= len(listUsers)
    print(accuracy)
    print('ENSEMBLE')



runTest('1-2')
runTest('3')
runTest('4-5')