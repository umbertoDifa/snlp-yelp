1)The zip Edinburgh contains the data already processed and POS-tagged from the Yelp Dataset:http://www.yelp.com/dataset_challenge

2)The code is in Python and can be run in the following way:

   -To prepare the edinburgh dataset run EdinburghPOSDiv.py

   -To prepare data for method1 run extractUsers.py
   -To classify using method1 run ensemblePerUser.py

   -To prepare data for method2 run extractUsersV2.py
   -To classify using method2 run SVMscriptV2.py and then unionOfClassifiers.py
---------------------------------------------------------
   -The errorAnalysis.py compute the error analysis
   -The baseline.py is the baseline on the all dataset
   -The baselineV2.py is the baseline per subset used in method2

   -The utility file contains several functions that save the preprocessed data
   -The trainAndTest.py spilts the dataset into train,test and validation
----------------------------------------------------------
   -All the other files represent attempt that have been discarded because did not perform as well as the final methods
	anyway they have been left in the folder for historic reasons