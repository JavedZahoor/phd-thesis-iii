from GlobalUtils import *
import ref_4_mifs as mifs
import ref_4_mi as mi
import sendemail as EMAIL
from sklearn.externals import joblib
import numpy
from MachineSpecificSettings import Settings
from DataSetLoaderLib import DataSetLoader
import time
@timing
def SelectSubSetmRMR(vectors, classes,useMethod,features):
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



d = DataSetLoader();
x = d.LoadDataSet("B_train");
y=d.LoadDataSetClasses("B_train");
print y.shape
y=numpy.transpose(y)
print x.shape
print y.shape
target=[]
y=list(y)
for i in y:
	target.append(int(i))
print len(y)
sizes=['10','50','100','150','200','250']
methods=['MRMR','JMI','JMIM']
for method in methods:
    for size in sizes:
	print size
        print method
	selected_indices=[]
	[subset,selected_indices] = SelectSubSetmRMR(x,target,method,int(size))
	joblib.dump(selected_indices,'datasetB_results/datasetB'+str(size)+'-'+method +'.joblib.pkl', compress=9) 
	print "Saved new selected indices"
	EMAIL.SendEmail(' DONE','datasetB'+str(size)+'-'+method +'.joblib.pkl')
