__author__ = 'vittorioselo'
import json
from collections import defaultdict

dataAllReview = []
dataTest = []
dataTrain = []

userCount = defaultdict(int)

with open('Edinburgh/ReviewsOfResutrantsEdinburghPOS.json') as inputFile:
    dataAllReview = json.load(inputFile)
    print(str(inputFile))

for line in dataAllReview:
    userCount['user_id'] += 1

for key in userCount.keys():
    userCount[key] = round(userCount[key] * 0.9)


for line in dataAllReview:
    if(userCount[line['user_id']]>= 0):
        userCount[line['user_id']] -= 1
        dataTrain.append(line)
    else:
        dataTest.append(line)

dataAllReview = None









