#%%
import importlib
import numpy as np
import sys
sys.path.append('Functions/')
import SMap_ridge as smr
import functions as fn
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
ts = np.loadtxt('input/deterministic_chaos_fc.txt')
jac = np.loadtxt('input/jacobian_chaos_fc.txt')
cross_validation_options = ['LOOCV', 'RollingCV']
cross_validation_type = cross_validation_options[0]
print('Cross validation method:', cross_validation_type)
length_training = 400

training_set = ts[0:length_training,:]
true_jacobian = jac[0:length_training,:]
### If you are running rolling window cross validation you need the unscale data
if cross_validation_type == 'RollingCV':
    unscaled_training_set = training_set
training_set, scaler = fn.scale_training_data(training_set)
parameters = ParameterGrid({'lambda': np.logspace(-5,-2,15),
                            'theta': np.logspace(-1,1.2,15)})



#%%
if cross_validation_type == 'LOOCV':
    ### Run leave one out cross validation to select the best hyperparameters
    print('Running:', cross_validation_type, '... This will take a while ...')
    e,l,t = cv.loocv(parameters, training_set, par = True, intercept=True)
    print(' ... done')
else:
    ### Or Rolling window cross validation:
    print('Running:', cross_validation_type, '... This will take a while ...')
    e,l,t = cv.rollingcv(parameters, unscaled_training_set, 20, par = True, intercept=True)
    print(' ... done')


#%%
#### In sample statistics
smap_object = smr.SMRidge(l,t)
c0, jacobians = smap_object.get_para(training_set, intercept=True)
train_pred = smap_object.fit(training_set,c0, jacobians)
rmse = smap_object.score(training_set[1:np.shape(training_set)[0],:],train_pred)
fig = plt.figure()
plt.plot(training_set[1:np.shape(training_set)[0],2], color = 'b')
plt.plot(train_pred[:,2], color = 'r')



#%%
#### Out-of-sample statistics
orizzonte = 40
sp = 0
pred = smap_object.predict(training_set,orizzonte, intercept = True)
### Scale back the prediction using the mean and standard deviation of the training set
pred = fn.unscale_test_data(pred, scaler)
test_data = ts[length_training:(length_training+orizzonte),:]
corr = np.mean([stat.pearsonr(pred[:,n],test_data[:,n])[0] 
                for n in range(np.shape(pred)[1])])
rmse_test = smap_object.score(test_data,pred)
fig = plt.figure()
plt.plot(test_data[:,sp], color = 'b')
plt.plot(pred[:,sp], color = 'r')
print('Out of sample correlation:', corr)
print('Out of sample rmse:', rmse_test )



#%%
infered_vcr = preprocessing.scale(fn.vcr(jacobians))
true_jacobian_matrix=fn.unfold_jacobian(true_jacobian,np.shape(ts)[1])
true_vcr = np.delete(preprocessing.scale(fn.vcr(true_jacobian_matrix)), 0)
plt.rcParams['figure.dpi']= 100
fig = plt.figure()
plt.title('scaled volume contraction rate')
plt.plot(true_vcr, color = 'b')
plt.plot(infered_vcr , color = 'r')
print('VCR inference quality:',
stat.pearsonr(true_vcr, infered_vcr)[0])
print(fn.inference_quality(jacobians, true_jacobian_matrix))

