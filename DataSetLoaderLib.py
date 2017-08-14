from GlobalUtils import *
import scipy
from MachineSpecificSettings import Settings
from sklearn.externals import joblib
import pickle;


class DataSetLoader(object):
    @timing
    def LoadDataSet(self, dataSetType):
        s = Settings();
        if dataSetType == "A":
		variables=numpy.array(joblib.load('DatasetA_Validation.joblib.pkl'))
		return variables;
            #mat = scipy.io.loadmat(s.getBasePath() + s.getInterimPath() + s.getDatasetAFileName());
            #return mat['G0'][:, 0:s.sampleSize()];
	elif dataSetType == "B_train":
		variables=numpy.array(joblib.load('DataSetBGSE24417MAQCIITraining_data.joblib.pkl'))
		return variables;
	elif dataSetType == "B_test":
		variables=numpy.array(joblib.load('DataSetBGSE24417MAQCIIValidation_data.joblib.pkl'))
		return variables;

        else:
		print "INVALID INPUT"
        	logWarning("HARD CODED VALUE from DataSetLoaderLib.LoadDataSet()");
        	return [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]];
    @timing
    def LoadDataSetClasses(self, dataSetType):
        s = Settings();
        if dataSetType == "A":
		variables=numpy.array(joblib.load('DatasetA_ValidationClasses.joblib.pkl'))
		return variables;
            #mat = scipy.io.loadmat(s.getBasePath() + s.getInterimPath() + s.getDatasetAFileName());
            #return mat['G0'][:, 0:s.sampleSize()];
	elif dataSetType == "B_train":
		variables=numpy.array(joblib.load('DataSetBGSE24417MAQCIITraining_targets.joblib.pkl'))
		return variables;
	elif dataSetType == "B_test":
		variables=numpy.array(joblib.load('DataSetBGSE24417MAQCIIValidation_targets.joblib.pkl'))
		return variables;

        else:
		print "INVALID INPUT"
        	logWarning("HARD CODED VALUE from DataSetLoaderLib.LoadDataSetClasses()");
        	return [0, 1, 1, 1, 0, 1];



##WE DONT USE ANYTHING BELOW THIS			
    def GetPartSize(self, dataSetType):
        if dataSetType == "A":
            ## HARDCODING
            logWarning("HARD CODED VALUE from DataSetLoaderLib.GetPartSize()");
            return 2510;#20080; #should be 1004004/36
            
    def CacheTopXPerPart(self, dataSetType):
        if dataSetType == "A":
            ## HARDCODING
            logWarning("HARD CODED VALUE from DataSetLoaderLib.CacheTopXPerPart()");
            return 10;#1000
    @timing        
    def LoadEnhancedDataSet(self, dataSetType):
        s = Settings();
        if dataSetType == "A":
            with open('objs.pickle.backup') as f:  #the file will need to be f.pickle
                return pickle.load(f)[1];#this will need to be changed to [0]
        else:
            logWarning("HARD CODED VALUE from DataSetLoaderLib.LoadDataSet()");
            return [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]];
            
    def GetClassLabels(self, dataSetType):
        if dataSetType == "A":
            ## HARDCODING
            logWarning("HARD CODED VALUE from DataSetLoaderLib.GetClassLabels()");
            return [0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
1	,
1	,
1	,
1	,
1	,
1	,
1	,
1	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
1	,
1	,
1	,
1	,
1	,
1	,
1	,
1	,
1	,
1	,
1	,
1	,
0	,
0	,
0	,
0	,
1	,
1	,
1	,
1	,
0	,
0	,
0	,
0	,
1	,
1	,
1	,
1	,
0	,
0	,
0	,
0	,
0	,
0	,
0	,
0	
];#1000