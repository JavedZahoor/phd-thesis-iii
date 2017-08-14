import numpy as np
from sklearn.ensemble import AdaBoostClassifier 
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from MachineSpecificSettings import Settings
from DataSetLoaderLib import DataSetLoader
from sklearn.externals import joblib
from genetic_selection import GeneticSelectionCV
from sklearn.neural_network import MLPClassifier


y=np.array(joblib.load('DatasetA_ValidationClasses.joblib.pkl'))
d = DataSetLoader();
X_original = d.LoadDataSet("A");

sizes=['10','50','100','150','200','250']
methods=['MRMR','JMI','JMIM']
for method in methods:
    for size in sizes:
	X=X_original
	indices= joblib.load(method+' PICKLES/selected_indices_'+method+'.joblib.pkl')
	X=np.array(X)[:,indices]
	indices= joblib.load(method+' PICKLES/'+size+'-'+method+'.joblib.pkl')
	X=np.array(X)[:,indices]
	f=open('genetic/'+method+'-'+size+'.txt','w')
        print size
        print method
	#MLP REMOVED DUE TO MEMORY CONSTRAINTS
        #DECISION TREE REMOVED DUE TO MEMORY CONSTRAINTS
        #Extra tree classifier removed due to memory constraints
	#Random tree classifer removed due to memory constraints


        print "svm.SVC"
        f.write("svm.SVC\n")
        estimator=svm.SVC()
        selector = GeneticSelectionCV(estimator,
                                      cv=10,
                                      verbose=1,
                                      scoring="accuracy",
                                      n_population=50,
                                      crossover_proba=0.5,
                                      mutation_proba=0.2,
                                      n_generations=40,
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,
                                      caching=True,
                                      n_jobs=-1)
        selector = selector.fit(X, y)
        f.write("Number of features selected: "+str(selector.n_features_))
        print "Number of features selected: ",selector.n_features_
        print "Selected features indexes are:",selector.support_
        f.write('\n=======================\n')
##        print "AdaBoostClassifier"
##        f.write("AdaBoostClassifier\n")
##        estimator= AdaBoostClassifier()
##        selector = GeneticSelectionCV(estimator,
##                                      cv=10,
##                                      verbose=1,
##                                      scoring="accuracy",
##                                      n_population=50,
##                                      crossover_proba=0.5,
##                                      mutation_proba=0.2,
##                                      n_generations=40,
##                                      crossover_independent_proba=0.5,
##                                      mutation_independent_proba=0.05,
##                                      tournament_size=3,
##                                      caching=True,
##                                      n_jobs=-1)
##        selector = selector.fit(X, y)
##        f.write("Number of features selected: "+str(selector.n_features_))
##        print "Number of features selected: ",selector.n_features_
##        print "Selected features indexes are:",selector.support_
##        f.write('\n=======================\n')