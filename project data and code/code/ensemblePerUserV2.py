__author__ = 'vittorioselo'

def ensemblePerUser(rank):
    import pandas
    import numpy
    from sklearn import svm
    from sklearn import metrics
    import os
    from os.path import join, isfile
    from collections import defaultdict
    from sklearn import linear_model
    from sklearn.ensemble import RandomForestClassifier
    from errorAnalysis import meanError,setError


    listUsers = list()
    myPath = 'trainPerRank/'

    listUsers = [f for f in os.listdir(str(myPath)) if isfile(join(myPath, f))]
    #listUsers.remove('.DS_Store')

    dictResults = defaultdict(float)

    averageError = float()
    errorSet = int()
    i=0
    for user in listUsers:
        print(i)
        i+=1
        #======READING TRAIN SET========
        dataTrain = numpy.array(pandas.read_csv('trainPerRank/'+user, header=None))
        trainRank = list(map(lambda x:x[0],numpy.array(pandas.read_csv('trainPerRank/'+rank+'/stars/'+user, header=None))))

         #=========READING VALIDATION SET =========
        dataValidation = numpy.array(pandas.read_csv('validationPerRank/'+user, header=None))
        validationRank = list(map(lambda x:x[0],numpy.array(pandas.read_csv('validationPerRank/'+rank+'/stars/'+user, header=None))))


        #============READING TEST SET ==========
        dataTest = numpy.array(pandas.read_csv('testPerRank/'+user, header=None))
        testRank = list(map(lambda x:x[0],numpy.array(pandas.read_csv('testPerRank/'+rank+'/stars/'+user, header=None))))

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
        logreg.class_weight = 'balanced'
        logreg.multi_class = 'ovr'
        logreg.fit(dataTrain, trainRank)

        #========RANDOM FOREST ==========
        #forest = RandomForestClassifier(n_estimators=400)
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
        averageError += meanError(prediction,testRank)
        errorSet += setError(prediction,testRank)


    accuracy = float()

    for user in dictResults.keys():
            accuracy += dictResults[user]
            #print bad users
            #if dictResults[user]==0:
                #print('BAD: '+user)
        #print('========')
        #print(user)
        #print(dictResults[user])

    accuracy /= len(listUsers)

    print(accuracy)
    print('=============ERROR ANALYSIS=========')
    print(averageError/len(listUsers))
    print(errorSet)

ensemblePerUser('1-2')
ensemblePerUser('3')
ensemblePerUser('4-5')