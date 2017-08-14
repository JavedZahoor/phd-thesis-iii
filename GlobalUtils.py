import numpy
import math
#import time
from timeit import default_timer as timer
import json
import traceback
import pymysql.cursors

"""GLOBALS"""
debugMessages = True
warnings = False
infoMessages = True
query = ''
logTimings=1
logToFile=1
start_time = timer()
logFileName = 'log.txt'
""" """
with open(logFileName, "w") as logfile:
    logfile.write("started at %s " % start_time);
    logfile.close();

def logDebug(msg):
    if debugMessages:
        print "DEBUG>>>>> " + str(msg);

def logWarning(msg):
    if warnings:
        print "WARNING===== " + str(msg);

def logInfo(msg):
    if infoMessages:
        print "INFO..... " + str(msg);

def isClassifier(something):
    if something.startswith('MLP') or something.startswith('Ada') or something.startswith('DT ') or something.startswith('Extra') or something.startswith('Random') or something.startswith('SVM'):
        return True; 
def isFeatureSelector(something):
    if something.startswith('MRMR') or something.startswith('JMI') or something.startswith('MI'):
        return True;  
def isEnsemble(something):
    if something.startswith('Ensemble'):
        return True; 
def isFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
def save_to_db(dataset, file, FSSize, technique, classifier, validation, accuracy, variance, timeTaken):
    try:
        print("save_to_db started.")
        query = "insert into results(dataset, fileName, size, FStechnique, classifier, validationTechnique, accuracy, variance, timeTaken) values('"+dataset+"','"+file+"',"+str(FSSize)+",'"+technique+"','"+classifier+"','"+validation+"',"+accuracy+","+variance+", " + str(timeTaken) + ");";
        print(query);
        #connection = pymysql.connect(host='111.68.102.118',user='root',password='Javed_phd_2016',db='mysql',           charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
        connection = pymysql.connect(host='localhost',user='root',password='Javed_phd_2016',db='L115680',           charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)
        cursor=connection.cursor()
        cursor.execute(query)
        connection.commit()
        #myData={ID:'1',DEVICEID: 'D0001',SENSORID: sensor_id,TIMESTAMP: ts,READING: status,DEVICETYPE: sensor}
        cursor.close()
        connection.close()
        print("save_to_db ending.")        
    except Exception as e:
        print(e)
"""UTILITY FUNCTIONS
def logDebug(tag, msg):
	if(logTimings):
		print((timer() - start_time))
		print("--- to " + tag + "---");
	if(logToFile):
		with open(logFileName, "a") as logfile:
			logfile.write("--- to " + tag + "---");
			logfile.write("%s" %(timer() - start_time));
			logfile.write("\n");
			logfile.close();
	print(msg)
END UTILITY FUNCTIONS"""

##http://stackoverflow.com/questions/5478351/python-time-measure-function
def timing(f):
    def wrap(*args):
        time1 = timer()
        #change this to timer to get more precise estimate of timings
        print("Calling %s" %f.func_name);
        ret = f(*args)
        time2 = timer()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap
    

    