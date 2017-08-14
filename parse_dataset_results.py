from GlobalUtils import *

dataset = 'B'; #'B', 'A'
processed = ''; #'', '_NP'
validation = '_loo';#'', '_loo'
file = 'dataset' + dataset + validation + processed + '.txt'; 
path = '/home/javedzahoor/research/mattia/code/jz/results'#'C:\Users\javed.zahoor\Dropbox\PhD\Research-2015-05-04\Research\Thesis III\Results';
fullName = path + '/' + file;
f = open(fullName, "r");
technique="";
accuracy="";
variance="";
FSSize=0;
timeTaken = 0;
classifier =  '';
nextIsTime = False;

i=-1;

while True:
    i = i+1;
    v = f.readline();
    if not v: break
    if i==0:
        save_to_db(dataset, file, FSSize, technique, classifier, validation.replace('_',"").upper() + 'CV', accuracy, variance, timeTaken);
        #query = "insert into results(dataset, fileName, size, FStechnique, classifier, validationTechnique, accuracy, variance, timeTaken) values('"+dataset+"','"+file+"',"+str(FSSize)+",'"+technique+"','"+classifier+"','"+'"+validation.upper() + 'CV'+"'+"',"+accuracy+","+variance+", " + str(timeTaken) + ");";
        if(FSSize>0):
            print(query);
        if(isInt(v.replace("\n",""))):
            FSSize = v.replace("\n","");
    if isFeatureSelector(v.replace("\n","")):
        technique = v.replace("\n","");
        
    if v.startswith('Accuracy'):
        i=-1;
        accuracy = v.replace("\n","");
        accuracy = accuracy.replace('Accuracy: ',"");
        accuracy = accuracy.replace('(',"");
        accuracy = accuracy.replace(')',"");
        #print(accuracy);
        parts = accuracy.split('+/- ');
        accuracy = parts[0];
        #print(accuracy);
        variance = parts[1];
        #print(variance);
    if isClassifier(v.replace("\n","")) or isEnsemble(v.replace("\n","")):
        classifier = v.replace("\n","");
        if isEnsemble(v.replace("\n","")): #ignore one line that says Training Ensemble
            f.readline();
        timeTaken = f.readline().replace("\n","").replace('It took:',"");
    
        
    
f.close();
print('done with reading the file...');