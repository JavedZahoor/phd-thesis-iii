from sklearn.model_selection import cross_val_score
#Used for storing and loading the trained classifier
from sklearn.externals import joblib
import numpy
from MachineSpecificSettings import Settings
import scipy.io
from DataSetLoaderLib import DataSetLoader
from sklearn.neural_network import MLPClassifier

from sklearn import metrics


sizes=['10','50','100','150','200','250']
methods=['MRMR','JMI','JMIM']
for method in methods:
    for size in sizes:
	print size
        print method
        import time
        targets=numpy.array(joblib.load('DatasetA_ValidationClasses.joblib.pkl'))
        d = DataSetLoader();
        result = d.LoadDataSet("A");
        indices= joblib.load(method+' PICKLES/selected_indices_'+method+'.joblib.pkl')
        result=numpy.array(result)[:,indices]
        indices= joblib.load(method+' PICKLES/'+size+'-'+method+'.joblib.pkl')
        result=result[:,indices]
        print "MLP logistic sgd"
        clf = MLPClassifier(activation='logistic',solver='sgd')

        start_time=time.time()
        scores = cross_val_score(clf, result, targets, cv=10)
        end_time=time.time()-start_time
        print end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf,'results/'+'MLP_logistic_sgd_'+method+'-'+size+'.joblib.pkl')

        from sklearn.ensemble import AdaBoostClassifier 
        print "AdaBoostClassifier"
        clf = AdaBoostClassifier()
        start_time=time.time()
        scores = cross_val_score(clf, result, targets, cv=10)
        end_time=time.time()-start_time
        print end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf,'results/'+'AdaBoostClassifier_'+method+'-'+size+'.joblib.pkl')

        from sklearn import tree
        print "DT classifier"
        clf = tree.DecisionTreeClassifier()

        start_time=time.time()
        scores = cross_val_score(clf, result, targets, cv=10)
        end_time=time.time()-start_time
        print end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf,'results/'+'DT_classifier_'+method+'-'+size+'.joblib.pkl')


        from sklearn.ensemble import ExtraTreesClassifier
        print "Extra tree classifier"
        clf = ExtraTreesClassifier()
        start_time=time.time()
        scores = cross_val_score(clf, result, targets, cv=10)
        end_time=time.time()-start_time
        print end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf,'results/'+'Extra_tree_classifier_'+method+'-'+size+'.joblib.pkl')

        from sklearn.ensemble import RandomForestClassifier
        print "Random Forest"
        clf = RandomForestClassifier()
        start_time=time.time()
        scores = cross_val_score(clf, result, targets, cv=10)
        end_time=time.time()-start_time
        print end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf,'results/'+'Random_Forest_'+method+'-'+size+'.joblib.pkl')
        from sklearn import svm
        print "SVM SVC"
        clf = svm.SVC()

        start_time=time.time()
        scores = cross_val_score(clf, result, targets, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	joblib.dump(clf,'results/'+'SVM_SVC_'+method+'-'+size+'.joblib.pkl')
        #import sendemail as EMAIL
        #EMAIL.SendEmail('Classifier trained','Trained for '+method+' on a feature set size of '+size)



