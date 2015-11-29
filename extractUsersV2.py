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
bestUsers = [user for user in list(usersCount.items()) if user[1]>=MIN_REVIEWS]

#===========compute the features for each user
print('COMPUTING FEATURES FOR USER')
res = {}
for i in range(len(bestUsers)):
    res[bestUsers[i][0]]= calculateUnigram(dataEdinburghPOS,'user_id',bestUsers[i][0])

#==========collect intersection of features
print('COLLECTING BEST FEATURES FOR USER')
userFeatures = defaultdict(lambda :defaultdict(list))
MAX_FEATURES = 100
for i in  range(len(bestUsers)):
    for rank in range(5):
        listOfCount=(list(map(lambda x: [x[0],log(x[1]['count'])+log(x[1]['regularity'])], res[bestUsers[i][0]][rank].items())))
        countSorted=sorted(listOfCount,key=operator.itemgetter(1),reverse=True)
        namesCount=set(list(map(operator.itemgetter(0),countSorted))[0:MAX_FEATURES if MAX_FEATURES<= len(countSorted) else len(countSorted)])
        userFeatures[bestUsers[i][0]][rank] = list(filter(lambda x: x[0] in namesCount, res[bestUsers[i][0]][rank].items()))

print('BUSINESS TIME')
#============get id of business
testBusinesses = list(set(list(map(lambda x: x['business_id'],dataTest))))
#===========compute the features for each business
resBusiness = {}
for i in range(len(testBusinesses)):
    resBusiness[testBusinesses[i]]=calculateUnigram(dataEdinburghPOS,'business_id',testBusinesses[i])



#============get id of business
trainBusinesses = list(set(list(map(lambda x: x['business_id'],dataTrain))))
#===========compute the features for each business
resTrainBusiness = {}
for i in range(len(trainBusinesses)):
    resTrainBusiness[trainBusinesses[i]]=calculateUnigram(dataEdinburghPOS,'business_id',trainBusinesses[i])

#============get id of business
validationBusinesses = list(set(list(map(lambda x: x['business_id'],dataValidation))))
#===========compute the features for each business
resValidationBusiness = {}
for i in range(len(validationBusinesses)):
    resValidationBusiness[validationBusinesses[i]]=calculateUnigram(dataEdinburghPOS,'business_id',validationBusinesses[i])

#============for each user, find in the test his business

print('SAVE DATA')


dfCreation2(dataTrain,'trainPerRank',bestUsers,userFeatures,resTrainBusiness)
dfCreation2(dataTest,'testPerRank',bestUsers,userFeatures,resBusiness)
dfCreation2(dataValidation, 'validationPerRank',bestUsers,userFeatures,resValidationBusiness)

