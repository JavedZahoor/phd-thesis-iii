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
        X_train= d.LoadDataSet("B_train");
        y_train = d.LoadDataSetClasses("B_train");
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

        indices= joblib.load('datasetB_pickles/datasetB'+size+'-'+method+'.joblib.pkl')
        X_train=X_train[:,indices]
        print "MLP logistic sgd"
        clf_mlp = make_pipeline(preprocessing.StandardScaler(), MLPClassifier(activation='logistic',solver='sgd'))
        start_time=time.time()
        scores = cross_val_score(clf_mlp,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_mlp,'datasetB_results/'+'MLP_logistic_sgd_'+method+'-'+size+'.joblib.pkl')

        from sklearn.ensemble import AdaBoostClassifier 
        print "AdaBoostClassifier"
        clf_ada = make_pipeline(preprocessing.StandardScaler(), AdaBoostClassifier())
        start_time=time.time()
        scores = cross_val_score(clf_ada,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_ada,'datasetB_results/'+'AdaBoostClassifier_'+method+'-'+size+'.joblib.pkl')

        from sklearn import tree
        print "DT classifier"
        clf_tree = make_pipeline(preprocessing.StandardScaler(),tree.DecisionTreeClassifier())
        start_time=time.time()
        scores = cross_val_score(clf_tree,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_tree,'datasetB_results/'+'DT_classifier_'+method+'-'+size+'.joblib.pkl')


        from sklearn.ensemble import ExtraTreesClassifier
        print "Extra tree classifier"
        clf_extra = make_pipeline(preprocessing.StandardScaler(),ExtraTreesClassifier())
        start_time=time.time()
        scores = cross_val_score(clf_extra,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_extra,'datasetB_results/'+'Extra_tree_classifier_'+method+'-'+size+'.joblib.pkl')

        from sklearn.ensemble import RandomForestClassifier
        print "Random Forest"
        clf_random = make_pipeline(preprocessing.StandardScaler(),RandomForestClassifier())
        start_time=time.time()
        scores = cross_val_score(clf_random,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_random,'datasetB_results/'+'Random_Forest_'+method+'-'+size+'.joblib.pkl')
        
	from sklearn import svm
        print "SVM SVC"
        clf_svm = make_pipeline(preprocessing.StandardScaler(),svm.SVC(probability=True))
        start_time=time.time()
        scores = cross_val_score(clf_svm,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf_svm,'datasetB_results/'+'SVM_SVC_'+method+'-'+size+'.joblib.pkl')
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
	joblib.dump(eclf1 ,'datasetB_results/'+'Ensemble_hard_'+method+'-'+size+'.joblib.pkl')
        
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
	joblib.dump(eclf1,'datasetB_results/'+'Ensemble_soft_'+method+'-'+size+'.joblib.pkl')


