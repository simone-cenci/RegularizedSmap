'''
Here I make predictions from an ensemble of models so that
I can compute uncertainty due to model selection
Things to do:

1) Compute the ensemble of Jacobian coefficients

'''

#%%
import importlib
import functions as fn
import numpy as np
import SMap_ridge as smr
import cv as cv
from sklearn.model_selection import ParameterGrid
import make_ts as mk
import matplotlib.pylab as plt
importlib.reload(fn)
importlib.reload(smr)
importlib.reload(cv)
importlib.reload(mk)
import scipy.stats as stat


#%%
ts, jac = mk.make_lv(200)

#%%
length_training = 400
training_set = ts[0:length_training,:]
training_set = training_set #+ np.random.uniform(-0.05,0.05, (np.shape(training_set)[0],
                            #                          np.shape(training_set)[1]))
true_jacobian = jac[0:length_training]
unscaled_training_set = training_set 
training_set, scaler = fn.scale_training_data(training_set)
parameters = ParameterGrid({'lambda': np.logspace(-3,0,15), 
                            'theta': np.logspace(-1,1.2,15)})



#%%
print('... This will take a while ...')
e,l,t = cv.ensemble_rollingcv(parameters, unscaled_training_set, 20)
print(' ... done')


#%%
#### In sample statistics
forecast = []
train_fit = []
jacobian_list = []
orizzonte = 40
for n in range(len(l)):
    smap_object = smr.SMRidge(l[n],t[n])
    ### Training set
    jacobians = smap_object.get_para(training_set)
    jacobian_list.append(jacobians)
    train_fit.append(smap_object.fit(training_set,jacobians))
    ### Test set
    pred = smap_object.predict(training_set,orizzonte)
    forecast.append(fn.unscale_test_data(pred, scaler))

test_data = ts[length_training:(length_training+orizzonte),:]

train_ens, train_err = fn.ensemble_forecast(train_fit,e)
pred, err = fn.ensemble_forecast(forecast,e)
corr_ensemble = np.mean([stat.pearsonr(pred[:,n],test_data[:,n])[0] 
                for n in range(np.shape(pred)[1])])
corr_single = np.mean([stat.pearsonr(forecast[1][:,n],test_data[:,n])[0] 
                for n in range(np.shape(forecast[1])[1])])
rmse_ensemble = smap_object.score(pred,test_data)
rmse_single = smap_object.score(forecast[1],test_data)
print('Out of sample correlation ensemble:', corr_ensemble)
print('Out of sample correlation single:', corr_single)
print('Out of sample rmse ensemble:', rmse_ensemble)
print('Out of sample rmse single:', rmse_single)
#%%
plt.rcParams['figure.dpi']= 300
sp = 0
fig = plt.figure(figsize=(5,5))
plt.plot(np.linspace(0,length_training-1,length_training-1), training_set[1:length_training,sp], color = 'b',
            label = 'Data') 
plt.plot(np.linspace(0,length_training-1,length_training-1), train_fit[1][:,sp], color = 'g', 
            label = 'Minimum error')
plt.plot(np.linspace(0,length_training-1,length_training-1), train_ens[:,sp], color = 'red',
            label = 'Ensemble')
plt.fill_between(np.linspace(0,length_training-1,length_training-1), train_ens[:,sp]-1.96*train_err[:,sp],
                                                     train_ens[:,sp]+1.96*train_err[:,sp],
                                                     alpha = 0.5,color= 'r')

fig = plt.figure()
plt.plot(np.linspace(0,orizzonte,orizzonte), test_data[:,sp], color = 'b',
            label = 'Data') 
plt.plot(np.linspace(0,orizzonte,orizzonte), forecast[1][:,sp], color = 'g', 
            label = 'Minimum error')
plt.plot(np.linspace(0,orizzonte,orizzonte), pred[:,sp], color = 'red',
            label = 'Ensemble')
plt.fill_between(np.linspace(0,orizzonte,orizzonte), pred[:,sp]-1.96*err[:,sp],
                                                     pred[:,sp]+1.96*err[:,sp],
                                                     alpha = 0.5,color= 'r')
plt.legend()                                        
plt.show()

