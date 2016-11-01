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
	feat_selector.ranking_

	# call transform() on X to filter it down to selected features
	X_filtered = feat_selector.transform(X)
	return X_filtered
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

print("")
print("")
print("")
print("")
targets = numpy.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1])
variables = None
d = DataSetLoader();
G = d.LoadDataSet("A");
variables = G[:,2:100];
"""
convert an array to csv
http://stackoverflow.com/questions/16482895/convert-a-numpy-array-to-a-csv-string-and-a-csv-string-back-to-a-numpy-array
targetsString = ','.join(['%d' % num for num in targets[0]])
variablesString = ','.join(['%.5f' % num for num in variables[0]])
numpy.fromstring(targetsString, sep=',')

load a csv to an array
http://stackoverflow.com/questions/13381815/python-csv-text-file-to-arrayi-j
"""
SelectSubSetmRMR(variables, targets)
