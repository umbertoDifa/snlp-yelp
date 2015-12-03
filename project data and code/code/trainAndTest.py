__author__ = 'vittorioselo'
import json
from collections import defaultdict

def splitTrainValidationAndTest():
    dataAllReview = []
    dataTest = []
    dataTrain = []
    dataValidation = list()

    userCount = defaultdict(int)
    userCount10Perc = defaultdict(int)
    path = 'Edinburgh/ReviewsOfbusinessEdinburgh.json' #'Edinburgh/ReviewsOfResutrantsEdinburgh.json'
    pathPOS = 'Edinburgh/ReviewsOfEdinburghPOS.json' # 'Edinburgh/ReviewsOfResutrantsEdinburghPOS.json'

    with open(path) as inputFile:
        dataAllReview = json.load(inputFile)
        print(str(inputFile))

    for line in dataAllReview:
        userCount[line['user_id']] += 1
    print(userCount)

    for key in userCount.keys():
        userCount10Perc[key] = round(userCount[key] * 0.1)
        userCount[key] = round(userCount[key] * 0.8)


    for line in dataAllReview:
        if(userCount[line['user_id']]> 0):
            userCount[line['user_id']] -= 1
            dataTrain.append(line)
        elif(userCount10Perc[line['user_id']]>0):
            userCount10Perc[line['user_id']] -= 1
            dataValidation.append(line)
        else:
            dataTest.append(line)

    #load json of edinburgh with POS
    with open(pathPOS) as inputFile:
        dataEdinburghPOS = json.load(inputFile)

    return dataTrain, dataTest, dataValidation,dataEdinburghPOS
#-------TEST---------
#A,B = splitTrainAndTest()
#print(len(A))
#print(len(B))








