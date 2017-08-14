import scipy.io as io
import numpy as np
import numpy.ma as ma
mat = io.loadmat("validation.mat")
#Targets
print mat
target= (mat['DataSetBGSE24417MAQCIIValidation'][:,0])
#Data
a= (mat['DataSetBGSE24417MAQCIIValidation'][:,1:])
a= np.where(np.isnan(a), ma.array(a, mask=np.isnan(a)).mean(axis=0), a)
io.savemat("DataSetBGSE24417MAQCIIValidation.mat",{'data':a,'target':target})


