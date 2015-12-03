from pandas.core.frame import _list_of_dict_to_arrays
from sympy.functions.special.bessel import besseli

__author__ = 'Umberto'

import json
from collections import defaultdict #has the extended version of dict
import operator #The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python.
                # For example, operator.add(x, y) is equivalent to the expression x+y
from utility import *
from trainAndTest import *

#=============create dictionary of how many reviews in Edinburgh for each user
print('SPLIT TRAIN TEST VALIDATION')
dataTrain,dataTest, dataValidation,dataEdinburghPOS = splitTrainValidationAndTest()
print(len(dataTrain))
print(len(dataTest))
print(len(dataValidation))
usersCount = defaultdict(int)
for rev in dataTrain:
    usersCount[rev['user_id']] += 1

#============pick all the user with more than MIN_REVIEWS
MIN_REVIEWS = 50
bestUsers = [user for user in list(usersCount.items()) if user[1]>=MIN_REVIEWS] #72

#===========compute the features for each user
print('COMPUTING FEATURES FOR USER')
res = {}
for i in range(len(bestUsers)):
    res[bestUsers[i][0]]= calculateFeatures(dataEdinburghPOS,'user_id',bestUsers[i][0])

#==========collect intersection of features
print('COLLECTING BEST FEATURES FOR USER')
userFeatures={}
for i in  range(len(bestUsers)):

    listOfCount=list(map(lambda x: [x[0],x[1]['count']], res[bestUsers[i][0]].items()))
    countSorted=sorted(listOfCount,key=operator.itemgetter(1),reverse=True)
    MAX_FEATURES_TO_INTERSECT = 1000 if len(listOfCount)<=1000 else len(listOfCount)
    MAX_FEATURES_TO_UNITE = 1000 if len(listOfCount)<=1000 else len(listOfCount)
    namesCount=set(list(map(operator.itemgetter(0),countSorted))[0:MAX_FEATURES_TO_INTERSECT])

    listOfRegularity=list(map(lambda x: [x[0],x[1]['regularity']], res[bestUsers[i][0]].items()))
    countRegularity=sorted(listOfRegularity,key=operator.itemgetter(1),reverse=True)
    namesRegularity=set(list(map(operator.itemgetter(0),countRegularity))[0:MAX_FEATURES_TO_INTERSECT])

    intersection = namesCount & namesRegularity
    len(intersection)

    listOfRelevance=list(map(lambda x: [x[0],x[1]['relevance']], res[bestUsers[i][0]].items()))
    countRelevance=sorted(listOfRelevance,key=operator.itemgetter(1),reverse=True)
    namesRelevance=set(list(map(operator.itemgetter(0),countRelevance))[0:MAX_FEATURES_TO_UNITE])

    union = intersection | namesRelevance
    len(union)
    userFeatures[bestUsers[i][0]] = list(filter(lambda x: x[0] in union, res[bestUsers[i][0]].items()))

print('BUSINESS TIME')
#============get id of business
testBusinesses = list(set(list(map(lambda x: x['business_id'],dataTest)))) #1001
#===========compute the features for each business
resBusiness = {}
for i in range(len(testBusinesses)):
    resBusiness[testBusinesses[i]]=calculateFeatures(dataEdinburghPOS,'business_id',testBusinesses[i])



#============get id of business
trainBusinesses = list(set(list(map(lambda x: x['business_id'],dataTrain))))#2670
#===========compute the features for each business
resTrainBusiness = {}
for i in range(len(trainBusinesses)):
    resTrainBusiness[trainBusinesses[i]]=calculateFeatures(dataEdinburghPOS,'business_id',trainBusinesses[i])

#============get id of business
validationBusinesses = list(set(list(map(lambda x: x['business_id'],dataValidation))))#905
#===========compute the features for each business
resValidationBusiness = {}
for i in range(len(validationBusinesses)):
    resValidationBusiness[validationBusinesses[i]]=calculateFeatures(dataEdinburghPOS,'business_id',validationBusinesses[i])

#============for each user, find in the test his business

print('SAVE DATA')


dfCreation(dataTrain,'trainAll',bestUsers,userFeatures,resTrainBusiness)
dfCreation(dataTest,'testAll',bestUsers,userFeatures,resBusiness)
dfCreation(dataValidation, 'validationAll',bestUsers,userFeatures,resValidationBusiness)

