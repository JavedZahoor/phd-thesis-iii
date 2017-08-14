from GlobalUtils import *
import ref_4_mifs as mifs
import ref_4_mi as mi
features=50
useMethod='MRMR'
print features
print useMethod
@timing
def SelectSubSetmRMR(vectors, classes):
	X = vectors
	y = classes
	# define MI_FS feature selection method
	feat_selector = mifs.MutualInformationFeatureSelector(method=useMethod,n_features=features)

	# find all relevant features
	feat_selector.fit(X, y)

	# check selected features
	print (feat_selector.support_)

	
	# check ranking of features
	print (feat_selector.ranking_)
	print (len(feat_selector.ranking_))
	selected_indices=feat_selector.ranking_

	# call transform() on X to filter it down to selected features
	X_filtered = feat_selector.transform(X)
	return [X_filtered,selected_indices]
"""
TEST PROGRAM
"""

import scipy
from MachineSpecificSettings import Settings
import scipy.io
import numpy
from DataSetLoaderLib import DataSetLoader
import csv
#Used for storing and loading the trained classifier
from sklearn.externals import joblib

print("")
print("")
print("")
print("")
#targets = numpy.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1])
targets=numpy.array(joblib.load('DatasetA_ValidationClasses.joblib.pkl'))
variables = None
d = DataSetLoader();
variables = d.LoadDataSet("A");

#variables = G[:,0:100];
indices= joblib.load('selected_indices_MRMR.joblib.pkl')
variables =numpy.array(variables )[:,indices]
#print variables .shape
#print len(variables ) 
variables =variables ;

"""
convert an array to csv
http://stackoverflow.com/questions/16482895/convert-a-numpy-array-to-a-csv-string-and-a-csv-string-back-to-a-numpy-array
targetsString = ','.join(['%d' % num for num in targets[0]])
variablesString = ','.join(['%.5f' % num for num in variables[0]])
numpy.fromstring(targetsString, sep=',')

load a csv to an array
http://stackoverflow.com/questions/13381815/python-csv-text-file-to-arrayi-j
"""
selected_indices=[]
[subset,selected_indices] = SelectSubSetmRMR(variables, targets)
print "Indices selected are:"
print len(selected_indices)
#print subset.shape
#Saving the selected_indices
joblib.dump(selected_indices,str(features)+'-'+useMethod+'.joblib.pkl', compress=9) 
print "Saved new selected indices"
import sendemail as EMAIL
EMAIL.SendEmail(useMethod+' DONE',str(features)+'-'+useMethod+'.joblib.pkl')


