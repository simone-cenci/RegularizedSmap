'''
Things to do:

1) Check why you don't get good reconstruction of the VCR 

'''

#%%
import importlib
import functions as fn
import numpy as np
import SMap_ridge as smr
import cv as cv
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
import make_ts as mk
import matplotlib.pylab as plt
importlib.reload(fn)
importlib.reload(smr)
importlib.reload(cv)
importlib.reload(mk)
import scipy.stats as stat


#%%
ts, jac = mk.make_hs(200)

#%%
cross_validation_options = ['LOOCV', 'RollingCV']
cross_validation_type = cross_validation_options[0]
print('Cross validation method:', cross_validation_type)
length_training = 400
training_set = ts[0:length_training,:]
true_jacobian = jac[1:length_training+1]
### If you are running rolling window cross validation you need the unscale data
if cross_validation_type == 'RollingCV':
    unscaled_training_set = training_set 
training_set, scaler = fn.scale_training_data(training_set)
parameters = ParameterGrid({'lambda': np.logspace(-3,0,15), 
                            'theta': np.logspace(-1,1.2,15)})



#%%
if cross_validation_type == 'LOOCV':
    ### Run leave one out cross validation to select the best hyperparameters
    print('Running:', cross_validation_type, '... This will take a while ...')
    e,l,t = cv.loocv(parameters, training_set, par = False)
    print(' ... done')
else:
    ### Or Rolling window cross validation:
    print('Running:', cross_validation_type, '... This will take a while ...')
    e,l,t = cv.rollingcv(parameters, unscaled_training_set, 20, par = False)
    print(' ... done')

#%%
#### In sample statistics
smap_object = smr.SMRidge(l,t)
jacobians = smap_object.get_para(training_set)
train_pred = smap_object.fit(training_set,jacobians)
rmse = smap_object.score(training_set[1:np.shape(training_set)[0],:],train_pred)
fig = plt.figure()
plt.plot(training_set[1:np.shape(training_set)[0],2], color = 'b')
plt.plot(train_pred[:,2], color = 'r')
plt.show()


#%%
#### Out-of-sample statistics
orizzonte = 40
sp = 0
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
infered_vcr = preprocessing.scale(fn.vcr(jacobians))
true_vcr = preprocessing.scale(fn.vcr(true_jacobian))
plt.rcParams['figure.dpi']= 300
fig = plt.figure()
plt.title('scaled volume contraction rate')
plt.plot(true_vcr, color = 'b')
plt.plot(infered_vcr , color = 'r')
plt.show()
print('VCR inference quality:', 
stat.pearsonr(np.delete(true_vcr, 1), infered_vcr)[0])
print('Correlation matrix of Jacobians:\n', fn.inference_quality(jacobians, true_jacobian))
