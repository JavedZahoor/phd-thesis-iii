from GlobalUtils import *
import ref_4_mifs as mifs
import ref_4_mi as mi
import time
import math

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
#variables = G[:,0:100];
for i in G[:,545756:545756+668]:
	for x in i:
		if math.isnan(x):
			print ("isNaN found...")
		if x==0:
			print x
print ("done, found no issues....")