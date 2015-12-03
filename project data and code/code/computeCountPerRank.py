__author__ = 'Umberto'

import json
from collections import defaultdict #has the extended version of dict
import operator #The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python.
                # For example, operator.add(x, y) is equivalent to the expression x+y
from utility import truncate
from statistics import mean
path = 'Edinburgh/ReviewsOfbusinessEdinburgh.json' #'Edinburgh/ReviewsOfResutrantsEdinburgh.json'

with open(path) as inputFile:
    dataEdinburgh = json.load(inputFile)

listOfStars = []
count = [0 for x in range(5)]
for rev in dataEdinburgh:
    count[truncate(rev['stars'])-1] +=1
    listOfStars.append(truncate(rev['stars']))

sum = sum(count)
for i in range(len(count)):
    count[i] = count[i]/sum
    print(i+1,'->',count[i])

import matplotlib.pyplot as plt
plt.bar(range(1,len(count)+1), count,align='center')
plt.title("Stars Frequency")
plt.xlabel("Stars")
plt.xticks(range(1,len(count)+1), ['1','2','3','4','5'])
plt.ylabel("Frequency")
plt.show()

print('Mean of stars distributions is ',mean(listOfStars))
#===================conteggio mezze stelle
count = [0 for x in range(4)]
for rev in dataEdinburgh:
    if rev['stars'] ==1.5 :
        count[0] +=1
    elif rev['stars'] == 2.5:
        count[1] +=1
    elif rev['stars'] == 3.5:
        count[2] +=1
    elif rev['stars'] == 4.5:
        count[3] +=1

for i in range(len(count)):
    print(i+1.5,'->',count[i])
