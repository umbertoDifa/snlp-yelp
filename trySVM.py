__author__ = 'vittorioselo'
import pandas
import numpy
from sklearn import svm
from sklearn import metrics
from sklearn.utils import column_or_1d


#name ="3dgTdRn62kJy94TGO24heA.csv"
name = 'Mjfc9jAPCFbIyW4Cmr11JA.csv'
dataTrain = numpy.array(pandas.read_csv('train/'+name, header=None))
#trainRank = numpy.array(pandas.read_csv('train/stars/'+name, header=None))

print(dataTrain[0])

#trainRank = [val for sublist in trainRank for val in sublist]
#trainRank = list(map(lambda x: int(x*5), trainRank))

#dataTest = numpy.array(pandas.read_csv('test/'+name, header=None))
#testRank = numpy.array(pandas.read_csv('test/stars/'+name, header=None))

#testRank = [val for sublist in testRank for val in sublist]
#testRank = list(map(lambda x: int(x*5), testRank))

#clf = svm.LinearSVC()
#clf.fit(dataTrain, trainRank)

#clf = svm.SVC()
#clf.decision_function_shape = 'ovr'
#clf.fit(dataTrain, trainRank)

#predicted = clf.predict(dataTest)

#print(predicted)
#print(testRank)

#print(metrics.accuracy_score(testRank,predicted))
#print(metrics.classification_report(testRank, predicted))
#print(metrics.confusion_matrix(testRank, predicted))



