__author__ = 'Umberto'

import json
from collections import defaultdict #has the extended version of dict
import operator #The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python.
                # For example, operator.add(x, y) is equivalent to the expression x+y
from utility import truncate

path = 'Edinburgh/ReviewsOfbusinessEdinburgh.json' #'Edinburgh/ReviewsOfResutrantsEdinburgh.json'

with open(path) as inputFile:
    dataEdinburgh = json.load(inputFile)

count = [0 for x in range(5)]
for rev in dataEdinburgh:
    count[truncate(rev['stars'])-1] +=1

sum = sum(count)
for i in range(len(count)):
    count[i] = count[i]/sum
    print(i+1,'->',count[i])

import matplotlib.pyplot as plt
plt.bar(range(len(count)), count)
plt.title("Stars Frequency")
plt.xlabel("Stars")
plt.ylabel("Frequency")
plt.show()
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
