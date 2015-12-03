__author__ = 'vittorioselo'

from collections import defaultdict

def meanError(prediction, correct):
    error = float()
    for i in range(len(correct)):
        error += (correct[i]-prediction[i])
    error /= len(correct)
    return error

def setError(prediction, correct):
    dict = defaultdict(list)
    error = int()
    for i in range(len(correct)):
        dict[correct[i]].append(i)
    for i in range(len(prediction)):
        flag = 1
        for y in range(prediction[i],5,1):
            if i in dict[prediction[y]]:
                flag = 0
        if flag == 1:
            error +=1
    return error

