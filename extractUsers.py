__author__ = 'Umberto'

import json
from collections import defaultdict #has the extended version of dict
import operator #The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python.
                # For example, operator.add(x, y) is equivalent to the expression x+y
from utility import calculateFeatures

#load plain json of edinburgh
with open('Edinburgh/ReviewsOfResutrantsEdinburgh.json') as inputFile:
    dataEdinburgh = json.load(inputFile)
#load json of edinburgh with POS

with open('Edinburgh/ReviewsOfResutrantsEdinburghPOS.json') as inputFile:
    dataEdinburghPOS = json.load(inputFile)

#=============create dictionary of how many reviews in Edinburgh for each user
usersCount = defaultdict(int)
for rev in dataEdinburgh:
    usersCount[rev['user_id']] += 1

#============pick all the user with more than MIN_REVIEWS
MIN_REVIEWS = 20
bestUsers = [user for user in list(usersCount.items()) if user[1]>=MIN_REVIEWS]

#===========compute the features for each user
res = []
for i in range(len(bestUsers)):
    res.append(calculateFeatures(dataEdinburghPOS,bestUsers[i][0]))

#==========collect intersection of features
userFeatures={}
for i in  range(len(bestUsers)):
    MAX_FEATURES_TO_INTERSECT = 30
    MAX_FEATURES_TO_UNITE = 15
    listOfCount=list(map(lambda x: [x[0],x[1]['count']], res[i].items()))
    countSorted=sorted(listOfCount,key=operator.itemgetter(1),reverse=True)
    namesCount=set(list(map(operator.itemgetter(0),countSorted))[0:MAX_FEATURES_TO_INTERSECT])

    listOfRegularity=list(map(lambda x: [x[0],x[1]['regularity']], res[i].items()))
    countRegularity=sorted(listOfRegularity,key=operator.itemgetter(1),reverse=True)
    namesRegularity=set(list(map(operator.itemgetter(0),countRegularity))[0:MAX_FEATURES_TO_INTERSECT])

    intersection = namesCount & namesRegularity
    len(intersection)

    listOfRelevance=list(map(lambda x: [x[0],x[1]['relevance']], res[i].items()))
    countRelevance=sorted(listOfRelevance,key=operator.itemgetter(1),reverse=True)
    namesRelevance=set(list(map(operator.itemgetter(0),countRelevance))[0:MAX_FEATURES_TO_UNITE])

    union = intersection | namesRelevance
    len(union)
    userFeatures[bestUsers[i][0]] = list(filter(lambda x: x[0] in union, res[i].items()))
