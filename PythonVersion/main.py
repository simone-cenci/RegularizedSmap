import os
os.chdir('/Users/simonecenci14/Desktop/Python_SMap/')

'''
Things to do:

1) Make the prediction function work properly
2) Check why you don't get good reconstruction of the VCR (double check the CR model)

'''


#%%
import importlib
import functions as fn
importlib.reload(fn)
import numpy as np
import SMap as sm
importlib.reload(sm)
from sklearn.model_selection import ParameterGrid
import make_ts as mk
importlib.reload(mk)
import matplotlib.pylab as plt
import scipy.stats as stat

#%%
ts, jac = mk.make_lv(200)

#%%
length_training = 400
training_set = ts[0:length_training,:]
true_jacobian = jac[0:length_training]
training_set, scaler = fn.scale_training_data(training_set)
parameters = ParameterGrid({'lambda': np.logspace(-3,0,15), 
                            'theta': np.logspace(-1,1.2,15)})


#%%
### Run leave one out cross validation to select the best hyperparameters
e,l,t = sm.loocv(parameters, training_set)

#%%
#### In sample statistics
smap_object = sm.SM(l,t)
jacobians = smap_object.get_para(training_set)
train_pred = smap_object.fit(training_set,jacobians)
rmse = smap_object.score(training_set[1:np.shape(training_set)[0],:],train_pred)
fig = plt.figure()
plt.plot(training_set[1:np.shape(training_set)[0],2], color = 'b')
plt.plot(train_pred[:,2], color = 'r')
plt.show()


#%%
#### Out-of-sample statistics
orizzonte = 20
sp = 2
pred = smap_object.predict(training_set,orizzonte)
### Scale back the prediction using the mean and standard deviation of the training set
pred = fn.unscale_test_data(pred, scaler)
test_data = ts[length_training:(length_training+orizzonte),:]
corr = np.mean([stat.pearsonr(pred[:,n],test_data[:,n])[0] 
                for n in range(np.shape(pred)[1])])
rmse_test = smap_object.score(test_data,pred)
fig = plt.figure()
plt.plot(test_data[:,sp], color = 'b')
plt.plot(pred[:,sp], color = 'r')
plt.show()
print('Out of sample correlation:', corr)
print('Out of sample rmse:', rmse_test )


#%%
infered_vcr = fn.vcr(jacobians)
true_vcr = fn.vcr(true_jacobian)

print('VCR inference quality:', 
stat.pearsonr(np.delete(true_vcr, np.shape(true_vcr)[0]-1), infered_vcr)[0])
fig = plt.figure()
plt.plot(true_vcr, color = 'b')
plt.plot(infered_vcr , color = 'r')
plt.show()

