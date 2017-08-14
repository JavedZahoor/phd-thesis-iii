#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

'''
Methods
get_params([deep]) 				Get parameters for this estimator.
predict(X) 						Predict class for X.
fit(X, y[, sample_weight]) 		Build a forest of trees from the training set (X, y).
set_params(\*\*params) 			Set the parameters of this estimator.


apply(X) 						Apply trees in the forest to X, return leaf indices.
decision_path(X) 				Return the decision path in the forest
fit_transform(X[, y]) 			Fit to data, then transform it.

predict_log_proba(X) 			Predict class log-probabilities for X.
predict_proba(X) 				Predict class probabilities for X.
score(X, y[, sample_weight]) 	Returns the mean accuracy on the given test data and labels.

'''
from sklearn.ensemble import RandomForestClassifier

import scipy
import scipy.io
import numpy
import time

from MachineSpecificSettings import Settings
from DataSetLoaderLib import DataSetLoader
#from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
#Used for storing and loading the trained classifier
from sklearn.externals import joblib

print("")
print("")
print("")
print("")

targets=numpy.array(joblib.load('DatasetA_ValidationClasses.joblib.pkl'))
d = DataSetLoader();
G = d.LoadDataSet("A");
indices= joblib.load('selected_indicesv2.joblib.pkl')
result=numpy.array(G)[:,indices]

randomForest = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None)

randomForest.fit(result, targets)
start_time=time.time()
scores = cross_val_score(randomForest, result, targets, cv=10)
end_time=time.time()-start_time
print "It took: ",end_time
for i in scores:
	print i
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

filename='randomForest_k-fold.joblib.pkl'
joblib.dump(randomForest,filename, compress=9) 

params = randomForest.get_params()

print params
''''''''''''''''''''''''''''''''''''''''''''''''

