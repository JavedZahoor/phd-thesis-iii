from sklearn.model_selection import cross_val_score
#Used for storing and loading the trained classifier
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
from MachineSpecificSettings import Settings
import scipy.io
from DataSetLoaderLib import DataSetLoader
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

curr_size=0
def create_model():
	model = Sequential()
	model.add(Dense(64, activation='elu', input_dim=curr_size))
	model.add(Dense(64, activation='elu'))
	model.add(Dense(64, activation='elu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='adamax',
              loss='binary_crossentropy',
              metrics=['accuracy'])
	return model


sizes=['10','50','100','150','200','250']
methods=['MRMR','JMI','JMIM']
for method in methods:
    for size in sizes:
	global curr_size
	curr_size=int(size)
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

        print "Keras"
	clf=make_pipeline(preprocessing.StandardScaler(), KerasClassifier(build_fn=create_model, epochs=10, batch_size=8, verbose=1))
	start_time=time.time()
        scores = cross_val_score(clf,X_train,  y_train, cv=10)
        end_time=time.time()-start_time
        print "It took: ",end_time
        for i in scores:
                print i
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

