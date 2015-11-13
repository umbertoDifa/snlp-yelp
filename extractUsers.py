__author__ = 'Umberto'

import json
from collections import defaultdict #has the extended version of dict
import operator #The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python.
                # For example, operator.add(x, y) is equivalent to the expression x+y
import calculateFeatures

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

#==========explore features
l=list(map(lambda x: [x[0],x[1]['count']], res[1].items()))
resSort=sorted(l,key=operator.itemgetter(1),reverse=True)
print(resSort[1:30])

resSort=sorted(l,key=operator.itemgetter(1),reverse=False)
print(resSort[1:30])


for r in dataEdinburgh:
    if(r['user_id']==bestUsers[1][0] and 'folk' in r['text']):
        print('-------------------------------------')
        print(r['text'])

