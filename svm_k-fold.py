from sklearn.model_selection import cross_val_score
#Used for storing and loading the trained classifier
from sklearn.externals import joblib
import numpy
from MachineSpecificSettings import Settings
import scipy.io
from DataSetLoaderLib import DataSetLoader
from sklearn import svm
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
clf = svm.SVC(kernel='poly')
import time
start_time=time.time()
scores = cross_val_score(clf, result, targets, cv=10)
end_time=time.time()-start_time
print "It took: ",end_time
for i in scores:
	print i
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

filename='svm.joblib.pkl'
joblib.dump(clf,filename, compress=9) 

params = clf.get_params()

print params