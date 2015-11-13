__author__ = 'vittorioselo'
import json
from collections import defaultdict

def splitTrainAndTest():
    dataAllReview = []
    dataTest = []
    dataTrain = []

    userCount = defaultdict(int)

    with open('Edinburgh/ReviewsOfResutrantsEdinburgh.json') as inputFile:
        dataAllReview = json.load(inputFile)
        print(str(inputFile))

    for line in dataAllReview:
        userCount[line['user_id']] += 1
    print('fanculo')
    print(userCount)

    for key in userCount.keys():
        userCount[key] = round(userCount[key] * 0.9)


    for line in dataAllReview:
        if(userCount[line['user_id']]>= 0):
            userCount[line['user_id']] -= 1
            dataTrain.append(line)
        else:
            dataTest.append(line)

    return dataTrain, dataTest
#-------TEST---------
#A,B = splitTrainAndTest()
#print(len(A))
#print(len(B))








