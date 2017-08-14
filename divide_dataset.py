from sklearn.externals import joblib
import numpy
from MachineSpecificSettings import Settings
import scipy.io
from DataSetLoaderLib import DataSetLoader
from sklearn.metrics import accuracy_score
import time

y=numpy.array(joblib.load('DatasetA_ValidationClasses.joblib.pkl'))
x = DataSetLoader();
x = x.LoadDataSet("A");

train_p=0
train_n=0
test_p=0
test_n=0
total=0
x_test=[]
y_test=[]
x_train=[]
y_train=[]
for i in range(0,len(y)):
    if y[i]==1:
        if train_p<26:
            x_train.append(x[i])
            y_train.append(y[i])
            train_p+=1
        if test_p<28:
            x_test.append(x[i])
            y_test.append(y[i])
            test_p+=1
            
    else:
        if train_n<44:
            x_train.append(x[i])
            y_train.append(y[i])
            train_n+=1
        if test_n<60:
            x_test.append(x[i])
            y_test.append(y[i])
            test_n+=1



#=====================
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
x_test_orig=x_test
x_train_orig=x_train

sizes=['10','50','100','150','200','250']
methods=['MRMR','JMI','JMIM']
for method in methods:
    for size in sizes:
	x_test=x_test_orig
	x_train=x_train_orig
	indices= joblib.load(method+' PICKLES/selected_indices_'+method+'.joblib.pkl')
	x_test=numpy.array(x_test)[:,indices]
	x_train=numpy.array(x_train)[:,indices]
	indices= joblib.load(method+' PICKLES/'+size+'-'+method+'.joblib.pkl')
	x_test=x_test[:,indices]
	x_train=x_train[:,indices]

	f=open('divided_results/'+method+'-'+size+'.txt','w')
        print size
        print method

        print "MLP logistic sgd"
	f.write("MLP logistic sgd\n")
        clf = MLPClassifier(activation='logistic',solver='sgd')
        start_time=time.time()
        print "Time it took to train: "
	f.write("Time it took to train:\n")
        clf.fit(x_train,y_train)
        end_time=time.time()-start_time
        print end_time
	f.write(str(end_time))
        y_pred=clf.predict(x_test)
	joblib.dump(clf,'divided_results/'+'MLP_logistic_sgd_'+method+'-'+size+'.joblib.pkl')
        print "Accuracy scores: "
	f.write("\nAccuracy scores:\n")
        print accuracy_score(y_test, y_pred)
	f.write(str(accuracy_score(y_test, y_pred)))
	f.write('\n=======================\n')

	print "AdaBoostClassifier"
	f.write("\nAdaBoostClassifier")
	clf = AdaBoostClassifier()
        start_time=time.time()
        print "Time it took to train: "
	f.write("Time it took to train:\n")
        clf.fit(x_train,y_train)
        end_time=time.time()-start_time
        print end_time
	f.write(str(end_time))
        y_pred=clf.predict(x_test)
	joblib.dump(clf,'divided_results/'+'AdaBoostClassifier_'+method+'-'+size+'.joblib.pkl')
        print "Accuracy scores: "
	f.write("\nAccuracy scores:\n")
        print accuracy_score(y_test, y_pred)
	f.write(str(accuracy_score(y_test, y_pred)))
	f.write('\n=======================\n')

	print "DT classifier"
	f.write("\nDT classifier")
	clf = tree.DecisionTreeClassifier()
	start_time=time.time()
        print "Time it took to train: "
	f.write("Time it took to train:\n")
        clf.fit(x_train,y_train)
        end_time=time.time()-start_time
        print end_time
	f.write(str(end_time))
        y_pred=clf.predict(x_test)
	joblib.dump(clf,'divided_results/'+'DT_classifier_'+method+'-'+size+'.joblib.pkl')
        print "\nAccuracy scores: "
	f.write("Accuracy scores:\n")
        print accuracy_score(y_test, y_pred)
	f.write(str(accuracy_score(y_test, y_pred)))
	f.write('\n=======================\n')
	
	f.write("\nExtra tree classifier")
	print "Extra tree classifier"
	clf = ExtraTreesClassifier()
	start_time=time.time()
        print "Time it took to train: "
	f.write("Time it took to train:\n")
        clf.fit(x_train,y_train)
        end_time=time.time()-start_time
        print end_time
	f.write(str(end_time))
        y_pred=clf.predict(x_test)
	joblib.dump(clf,'divided_results/'+'Extra_tree_classifier_'+method+'-'+size+'.joblib.pkl')
        print "\nAccuracy scores: "
	f.write("Accuracy scores:\n")
        print accuracy_score(y_test, y_pred)
	f.write(str(accuracy_score(y_test, y_pred)))
	f.write('\n=======================\n')

	f.write("\nRandom Forest")
	print "Random Forest"
	clf = RandomForestClassifier()
	start_time=time.time()
        print "Time it took to train: "
	f.write("Time it took to train:\n")
        clf.fit(x_train,y_train)
        end_time=time.time()-start_time
        print end_time
	f.write(str(end_time))
        y_pred=clf.predict(x_test)
	joblib.dump(clf,'divided_results/'+'Random_Forest_'+method+'-'+size+'.joblib.pkl')
        print "Accuracy scores: "
	f.write("\nAccuracy scores:\n")
        print accuracy_score(y_test, y_pred)
	f.write(str(accuracy_score(y_test, y_pred)))
	f.write('\n=======================\n')

	f.write("\nSVM SVC")
	print "SVM SVC"
	clf = svm.SVC()
	start_time=time.time()
        print "Time it took to train: "
	f.write("Time it took to train:\n")
        clf.fit(x_train,y_train)
        end_time=time.time()-start_time
        print end_time
	f.write(str(end_time))
        y_pred=clf.predict(x_test)
	joblib.dump(clf,'divided_results/'+'SVM_SVC_'+method+'-'+size+'.joblib.pkl')
        print "Accuracy scores: "
	f.write("\nAccuracy scores:\n")
        print accuracy_score(y_test, y_pred)
	f.write(str(accuracy_score(y_test, y_pred)))
	f.write('\n=======================\n')
	f.close()


