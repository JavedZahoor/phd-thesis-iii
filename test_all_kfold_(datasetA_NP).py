from sklearn.model_selection import cross_val_score
#Used for storing and loading the trained classifier
from sklearn.externals import joblib
import numpy
from MachineSpecificSettings import Settings
import scipy.io
from DataSetLoaderLib import DataSetLoader
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier

sizes=['10','50','100','150','200','250']
methods=['MRMR','JMI','JMIM']
for method in methods:
    for size in sizes:
	print size
        print method
        import time
	d = DataSetLoader();
        X_train= d.LoadDataSet("A");
        y_train = d.LoadDataSetClasses("A");
	print X_train.shape
	print y_train.shape
	#chaipee will fix it later on
	y_train=numpy.transpose(y_train)
	print y_train.shape
	targets=list(y_train)
	y_train=[]
	for i in targets:
		#print i
		y_train.append(int(i))
	#print len(y_train)
	#first run indices
	indices= joblib.load('datasetA_pickles/selected_indices_'+method+'.joblib.pkl')
	X_train=X_train[:,indices]
	#second run indices
        indices= joblib.load('datasetA_pickles/'+size+'-'+method+'.joblib.pkl')
        X_train=X_train[:,indices]

        print "MLP logistic sgd"
	clf_mlp = MLPClassifier(activation='logistic',solver='sgd')
        start_time=time.time()
        scores = cross_val_score(clf_mlp,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_mlp,'datasetA_results/'+'MLP_logistic_sgd_'+method+'-'+size+'.joblib.pkl')

        from sklearn.ensemble import AdaBoostClassifier 
        print "AdaBoostClassifier"
	clf_ada = AdaBoostClassifier()
        start_time=time.time()
        scores = cross_val_score(clf_ada,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_ada,'datasetA_results/'+'AdaBoostClassifier_'+method+'-'+size+'.joblib.pkl')

        from sklearn import tree
        print "DT classifier"
	clf_tree = tree.DecisionTreeClassifier()
        start_time=time.time()
        scores = cross_val_score(clf_tree,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_tree,'datasetA_results/'+'DT_classifier_'+method+'-'+size+'.joblib.pkl')


        from sklearn.ensemble import ExtraTreesClassifier
        print "Extra tree classifier"
	clf_extra = ExtraTreesClassifier()
        start_time=time.time()
        scores = cross_val_score(clf_extra,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_extra,'datasetA_results/'+'Extra_tree_classifier_'+method+'-'+size+'.joblib.pkl')

        from sklearn.ensemble import RandomForestClassifier
        print "Random Forest"
	clf_random = RandomForestClassifier()
        start_time=time.time()
        scores = cross_val_score(clf_random,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_random,'datasetA_results/'+'Random_Forest_'+method+'-'+size+'.joblib.pkl')
        
	from sklearn import svm
        print "SVM SVC"
	clf_svm = svm.SVC( probability=True)
        start_time=time.time()
        scores = cross_val_score(clf_svm,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_svm,'datasetA_results/'+'SVM_SVC_'+method+'-'+size+'.joblib.pkl')
        #import sendemail as EMAIL
        #EMAIL.SendEmail('Classifier trained','Trained for '+method+' on a feature set size of '+size)

	print "Ensemble hard classifier"
	eclf1 = VotingClassifier(estimators=[('svm', clf_svm), ('rf', clf_random), ('et', clf_extra), ('tree', clf_tree),('mlp', clf_mlp),('ada', clf_ada)], voting='hard')
	print "Training classifier"
        start_time=time.time()
	scores = cross_val_score(eclf1,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(eclf1 ,'datasetA_results/'+'Ensemble_hard_'+method+'-'+size+'.joblib.pkl')
        
	print "Ensemble soft classifier"
	eclf1 = VotingClassifier(estimators=[('svm', clf_svm), ('rf', clf_random), ('et', clf_extra), ('tree', clf_tree),('mlp', clf_mlp),('ada', clf_ada)], voting='soft')
	print "Training classifier"
        start_time=time.time()
	scores = cross_val_score(eclf1,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(eclf1,'datasetA_results/'+'Ensemble_soft_'+method+'-'+size+'.joblib.pkl')
        


