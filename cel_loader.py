from Bio.Affy import CelFile
import os
import copy
from pylab import plot,show
from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten
general=[]
files = [f for f in os.listdir('.') if os.path.isfile(f)]
cel_files=[]
for f in files:
    if f.rfind(".CEL")!=-1:
        cel_files.append(f)
print cel_files
temp=0
#Proj_save = open("without_classification.txt", "w")
for i in cel_files:
    temp+=1
    print "Working with file number: "+str(temp)
    with open(i) as handle:
        c = CelFile.read(handle)
    arr=[]
    try:
        for i in c.intensities:
            for x in i:
                arr.append((x))
        #for i in arr:
        #    Proj_save.write(i+ ";")
        #Proj_save.write("\n")
        general.append(copy.deepcopy(arr))
    except:
        print "Skipped file(unreadable)"
#Proj_save.close()
print "Starting K mean"
data=whiten(array(general))
# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(data,2)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()