__author__ = 'Umberto'

from baseline import baseline
from unionOfClassifiers import runMethod2
from SVMScript import runMethod1

predictionsFromBaseline = baseline()
predictionsFromMethod2 = runMethod2()
predictionsFromMethod1 = runMethod1()

#flatten predictions
predictionsFromBaseline = [val for sublist in predictionsFromBaseline for val in sublist]
predictionsFromMethod2 = [val for sublist in predictionsFromMethod2 for val in sublist]
predictionsFromMethod1 = [val for sublist in predictionsFromMethod1 for val in sublist]

from scipy import stats
if(len(predictionsFromBaseline) != len(predictionsFromMethod2)):
    print('Error predictions from method 2 have different lengths!')
else:

    r1 = stats.ttest_ind(predictionsFromBaseline, predictionsFromMethod2)
    print(r1)
    r2 = stats.ttest_ind(predictionsFromBaseline, predictionsFromMethod2, equal_var = False)
    print(r2)
#
# (8.6566243900008022, 8.3173929492649013e-18)
# (8.6566243900008022, 1.2765689781551307e-17)

if(len(predictionsFromBaseline) != len(predictionsFromMethod1)):
    print('Error predictions from method 1 have different lengths!')
else:

    r1 = stats.ttest_ind(predictionsFromBaseline, predictionsFromMethod1)
    print(r1)
    r2 = stats.ttest_ind(predictionsFromBaseline, predictionsFromMethod1, equal_var = False)
    print(r2)

#
# (0.38934989764282057, 0.6970488281708882)
# (0.38934989764282057, 0.69704943709052203)