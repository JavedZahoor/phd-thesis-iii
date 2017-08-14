from sklearn.model_selection import cross_val_score
#Used for storing and loading the trained classifier
from sklearn.externals import joblib
import numpy
from MachineSpecificSettings import Settings
import scipy.io
from DataSetLoaderLib import DataSetLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

print("")
print("")
print("")
print("")

targets=numpy.array(joblib.load('DatasetA_ValidationClasses.joblib.pkl'))
d = DataSetLoader();
G = d.LoadDataSet("A");
indices= joblib.load('selected_indicesv2.joblib.pkl')
result=numpy.array(G)[:,indices]
clf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None)
import time
start_time=time.time()
scores = cross_val_score(clf, result, targets, cv=10)
end_time=time.time()-start_time
print end_time
for i in scores:
	print i
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


filename='rf_k-fold.joblib.pkl'
joblib.dump(clf,filename, compress=9) 

params = clf.get_params()

print params