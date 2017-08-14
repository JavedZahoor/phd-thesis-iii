import threading
import time
import ref_4_mifs as mifs
from sklearn.externals import joblib
def SelectSubSetmRMR(vectors, classes):
	X = vectors
	y = classes
	# define MI_FS feature selection method
	feat_selector = mifs.MutualInformationFeatureSelector()

	# find all relevant features
	feat_selector.fit(X, y)

	# check selected features
	feat_selector.support_

	# check ranking of features
	return feat_selector.ranking_

"""
TEST PROGRAM
"""
from GlobalUtils import *
import scipy
from MachineSpecificSettings import Settings
import scipy.io
import numpy
from DataSetLoaderLib import DataSetLoader
import csv

#variables =joblib.load('DatasetA_Validation.joblib.pkl')
#targets =joblib.load('DatasetA_ValidationClasses.joblib.pkl')
d = DataSetLoader();
variables  = d.LoadDataSet("A");
targets = d.LoadDataSetClasses("A");
def test_range(start,end,G,targets,half):
    try:
	print "first"
	print start
	print end-half
        SelectSubSetmRMR(G[:,start:end-half],targets)
    except:
	
	return 1
    try:
	print "second"
	print start+half
	print end
        SelectSubSetmRMR(G[:,start+half:end],targets)
    except:
	return 2
    return 3



vals=545756
original=668
start=vals-original
half=original/2
Error=1
variables=numpy.asarray(variables)
print variables.shape
G=numpy.delete(variables,545089,1)

while Error!=2:
     print half
     test=test_range(start,vals,G,targets,half)
     if test==1:
	vals=vals-half
	half=int((vals-start)/2)
	continue
     if test==2:
	start=start+half
	half=int((vals-start)/2)
	continue

print "Exiting Main Thread"
