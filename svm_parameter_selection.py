import numpy as np
import random
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from MachineSpecificSettings import Settings
from DataSetLoaderLib import DataSetLoader
from sklearn.externals import joblib
from evolutionary_search import EvolutionaryAlgorithmSearchCV


y=np.array(joblib.load('DatasetA_ValidationClasses.joblib.pkl'))
d = DataSetLoader();
X_original = d.LoadDataSet("A");
paramgrid = {"kernel": ["rbf"],
             "C"     : np.logspace(-9, 9, num=25, base=10),
             "gamma" : np.logspace(-9, 9, num=25, base=10)}

sizes=['10','50','100','150','200','250']
methods=['MRMR','JMI','JMIM']
targets=np.array(joblib.load('DatasetA_ValidationClasses.joblib.pkl'))
	
for method in methods:
    for size in sizes:
	random.seed(1)
	X=X_original
	indices= joblib.load(method+' PICKLES/selected_indices_'+method+'.joblib.pkl')
	X=np.array(X)[:,indices]
	indices= joblib.load(method+' PICKLES/'+size+'-'+method+'.joblib.pkl')
	X=np.array(X)[:,indices]
	f=open('genetic/'+method+'-'+size+'.txt','w')
        print size
        print method
        print "svm.SVC"
        f.write("svm.SVC\n")
        cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                   params=paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(targets, n_folds=10),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=-1)
	cv.fit(X, targets)
        f.write('\n=======================\n')
