from sklearn.model_selection import cross_val_score
#Used for storing and loading the trained classifier
from sklearn.externals import joblib
import numpy
from MachineSpecificSettings import Settings
import scipy.io
from DataSetLoaderLib import DataSetLoader
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
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
	X_test= d.LoadDataSet("B_test");
        y_test = d.LoadDataSetClasses("B_test");
	print X_test.shape
	print y_test.shape
	#chaipee will fix it later on
	y_train=numpy.transpose(y_train)
	y_test=numpy.transpose(y_test)
	print y_train.shape
	targets=list(y_train)
	test_targets=list(y_test)
	y_train=[]
	y_test=[]
	for i in targets:
		#print i
		y_train.append(int(i))
	for i in test_targets:
		y_test.append(int(i))
	#print len(y_train)

        indices= joblib.load('datasetB_pickles/datasetB'+size+'-'+method+'.joblib.pkl')
        X_train=X_train[:,indices]
	X_test=X_test[:,indices]


        print "MLP logistic sgd"
        clf_mlp = MLPClassifier(activation='logistic',solver='sgd')
	print "Training classifier"
        start_time=time.time()
	clf_mlp.fit(X_train,y_train)      
        end_time=time.time()-start_time
        print end_time
	print "Evaluation time"
	start_time=time.time()
	predictions=clf_mlp.predict(X_test)
	end_time=time.time()-start_time
        print end_time
        print(classification_report(y_test, predictions))
	joblib.dump(clf_mlp,'datasetB_results/'+'MLP_logistic_sgd_'+method+'-'+size+'.joblib.pkl')

        from sklearn.ensemble import AdaBoostClassifier 
        print "AdaBoostClassifier"
        clf_ada = AdaBoostClassifier()
        print "Training classifier"
        start_time=time.time()
	clf_ada.fit(X_train,y_train)      
        end_time=time.time()-start_time
        print end_time
	print "Evaluation time"
	start_time=time.time()
	predictions=clf_ada.predict(X_test)
	end_time=time.time()-start_time
        print end_time
        print(classification_report(y_test, predictions))
	joblib.dump(clf_ada,'datasetB_results/'+'AdaBoostClassifier_'+method+'-'+size+'.joblib.pkl')

        from sklearn import tree
        print "DT classifier"
        clf_tree = tree.DecisionTreeClassifier()
 	print "Training classifier"
        start_time=time.time()
	clf_tree.fit(X_train,y_train)      
        end_time=time.time()-start_time
        print end_time
	print "Evaluation time"
	start_time=time.time()
	predictions=clf_tree.predict(X_test)
	end_time=time.time()-start_time
        print end_time
        print(classification_report(y_test, predictions))
	joblib.dump(clf_tree,'datasetB_results/'+'DT_classifier_'+method+'-'+size+'.joblib.pkl')


        from sklearn.ensemble import ExtraTreesClassifier
        print "Extra tree classifier"
        clf_extra = ExtraTreesClassifier()
 	print "Training classifier"
        start_time=time.time()
	clf_extra.fit(X_train,y_train)      
        end_time=time.time()-start_time
        print end_time
	print "Evaluation time"
	start_time=time.time()
	predictions=clf_extra.predict(X_test)
	end_time=time.time()-start_time
        print end_time
        print(classification_report(y_test, predictions))
	joblib.dump(clf_extra,'datasetB_results/'+'Extra_tree_classifier_'+method+'-'+size+'.joblib.pkl')

        from sklearn.ensemble import RandomForestClassifier
        print "Random Forest"
        clf_random = RandomForestClassifier()
        print "Training classifier"
        start_time=time.time()
	clf_random.fit(X_train,y_train)      
        end_time=time.time()-start_time
        print end_time
	print "Evaluation time"
	start_time=time.time()
	predictions=clf_random.predict(X_test)
	end_time=time.time()-start_time
        print end_time
        print(classification_report(y_test, predictions))
	joblib.dump(clf_random,'datasetB_results/'+'Random_Forest_'+method+'-'+size+'.joblib.pkl')
        

	from sklearn import svm
        print "SVM SVC"
        clf_svm = svm.SVC(probability=True)

        print "Training classifier"
        start_time=time.time()
	clf_svm.fit(X_train,y_train)      
        end_time=time.time()-start_time
        print end_time
	print "Evaluation time"
	start_time=time.time()
	predictions=clf_svm.predict(X_test)
	end_time=time.time()-start_time
        print end_time
        print(classification_report(y_test, predictions))
	joblib.dump(clf_svm,'datasetB_results/'+'SVM_SVC_'+method+'-'+size+'.joblib.pkl')
        #import sendemail as EMAIL
        #EMAIL.SendEmail('Classifier trained','Trained for '+method+' on a feature set size of '+size)
	
	print "Ensemble hard classifier"
	eclf1 = VotingClassifier(estimators=[('svm', clf_svm), ('rf', clf_random), ('et', clf_extra), ('tree', clf_tree),('mlp', clf_mlp),('ada', clf_ada)], voting='hard')
	print "Training classifier"
        start_time=time.time()
	eclf1 .fit(X_train,y_train)      
        end_time=time.time()-start_time
        print end_time
	print "Evaluation time"
	start_time=time.time()
	predictions=eclf1 .predict(X_test)
	end_time=time.time()-start_time
        print end_time
        print(classification_report(y_test, predictions))
	joblib.dump(clf_svm,'datasetB_results/'+'Ensemble_hard_'+method+'-'+size+'.joblib.pkl')
        
	print "Ensemble soft classifier"
	eclf1 = VotingClassifier(estimators=[('svm', clf_svm), ('rf', clf_random), ('et', clf_extra), ('tree', clf_tree),('mlp', clf_mlp),('ada', clf_ada)], voting='soft')
	print "Training classifier"
        start_time=time.time()
	eclf1 .fit(X_train,y_train)      
        end_time=time.time()-start_time
        print end_time
	print "Evaluation time"
	start_time=time.time()
	predictions=eclf1 .predict(X_test)
	end_time=time.time()-start_time
        print end_time
        print(classification_report(y_test, predictions))
	joblib.dump(clf_svm,'datasetB_results/'+'Ensemble_soft_'+method+'-'+size+'.joblib.pkl')
        
