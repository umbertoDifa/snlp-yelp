__author__ = 'Umberto'

import json
from collections import defaultdict #has the extended version of dict
import operator #The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python.
                # For example, operator.add(x, y) is equivalent to the expression x+y
import pandas as pn
from utility import calculateFeatures
from trainAndTest import splitTrainAndTest

#load plain json of edinburgh
with open('Edinburgh/ReviewsOfResutrantsEdinburgh.json') as inputFile:
    dataEdinburgh = json.load(inputFile)
#load json of edinburgh with POS

with open('Edinburgh/ReviewsOfResutrantsEdinburghPOS.json') as inputFile:
    dataEdinburghPOS = json.load(inputFile)

#=============create dictionary of how many reviews in Edinburgh for each user
dataTrain,dataTest = splitTrainAndTest()
usersCount = defaultdict(int)
for rev in dataTrain:
    usersCount[rev['user_id']] += 1

#============pick all the user with more than MIN_REVIEWS
MIN_REVIEWS = 20
bestUsers = [user for user in list(usersCount.items()) if user[1]>=MIN_REVIEWS]

#===========compute the features for each user
res = {}
for i in range(len(bestUsers)):
    res[bestUsers[i][0]]= calculateFeatures(dataEdinburghPOS,'user_id',bestUsers[i][0])

#==========collect intersection of features
userFeatures={}
for i in  range(len(bestUsers)):
    MAX_FEATURES_TO_INTERSECT = 30
    MAX_FEATURES_TO_UNITE = 15
    listOfCount=list(map(lambda x: [x[0],x[1]['count']], res[bestUsers[i][0]].items()))
    countSorted=sorted(listOfCount,key=operator.itemgetter(1),reverse=True)
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


#============get id of business
testBusinesses = list(set(list(map(lambda x: x['business_id'],dataTest))))
#===========compute the features for each business
resBusiness = {}
for i in range(len(testBusinesses)):
    resBusiness[testBusinesses[i]]=calculateFeatures(dataEdinburghPOS,'business_id',testBusinesses[i])

#============for each user, find in the test his business

for i in range(len(bestUsers)):
    for rev in dataTest:
        if rev['user_id']==bestUsers[i][0]:
            bestWordsOfUser=list(map(lambda x: x[0], userFeatures[bestUsers[i][0]]))
            common = list(filter(lambda x: x in bestWordsOfUser,resBusiness[rev['business_id']].keys()))
            #common = set(resBusiness[rev['business_id']]) & set(res[bestUsers[i][0]])
            userValuesDictionary=list(map(lambda x: x[1], userFeatures[bestUsers[i][0]]))

            tmp = list()
            for v in userValuesDictionary:
                tmp.append(list(v.values()))

            for w in bestWordsOfUser:
                if w in common:
                    tmp.append(list(resBusiness[rev['business_id']][w].values()))
                else:
                    tmp.append([0,0,0])

             #unflatten
            tmp = [val for sublist in tmp for val in sublist]
            #scale perche ci piace
            tmp = list(map(lambda x: x*10, tmp))

            print(common)
            print(len(common))
#============for each business