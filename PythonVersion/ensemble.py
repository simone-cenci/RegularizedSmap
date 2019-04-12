#%%
import importlib
import numpy as np
import sys
sys.path.append('Functions/')
import SMap_ridge as smr
import functions as fn
import landscape as ld
import cv as cv
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
import matplotlib.pylab as plt
importlib.reload(fn)
importlib.reload(smr)
importlib.reload(cv)
importlib.reload(ld)
import scipy.stats as stat


#%%
length_training = 400
ts = np.loadtxt('input/deterministic_chaos_cr.txt')
analytical_jacobian = np.loadtxt('input/jacobian_chaos_cr.txt')
#ts = ts+ts*np.random.normal(0.,0.05*np.std(ts), size =np.shape(ts))
training_set = ts[0:length_training,:]
analytical_jacobian = analytical_jacobian[0:length_training,:]
unscaled_training_set = training_set
training_set, scaler = fn.scale_training_data(training_set)
parameters = ParameterGrid({'lambda': np.logspace(-5,0,15),
                            'theta': np.logspace(-1,1.2,15)})



#%%
print('... This will take a while ...')
full_error, e,l,t = cv.loocv(parameters, training_set, par = True, intercept = True, ensemble = True)
#full_error, e,l,t = cv.rollingcv(parameters, unscaled_training_set, 20, par = True, intercept = True, ensemble = True)
print(' ... done')

#%%
lm = np.array([parameters[k]['lambda'] for k in range(len(full_error))])
th = np.array([parameters[k]['theta'] for k in range(len(full_error))])
ld.landscape(np.column_stack([lm,th,full_error]))


#%%
orizzonte = 40
### Now make training ensemble, prediction_ensemble and jacobians in the ensemble
train_fit, train_ens, train_err, forecast, cv_forecast, pred, err, \
   cv_vcr, jacobian_list, jac_ens, jac_err = \
   fn.ensemble_method(training_set, l, t, orizzonte, scaler, e)

test_data = ts[length_training:(length_training+orizzonte),:]
corr_ensemble = np.mean([stat.pearsonr(pred[:,n],test_data[:,n])[0]
                for n in range(np.shape(pred)[1])])
corr_single = np.mean([stat.pearsonr(cv_forecast[:,n],test_data[:,n])[0]
                for n in range(np.shape(cv_forecast)[1])])
rmse_ensemble = fn.rmse(pred,test_data)
rmse_single = fn.rmse(cv_forecast,test_data)
print('Out of sample correlation ensemble:', corr_ensemble)
print('Out of sample correlation single:', corr_single)
print('Out of sample rmse ensemble:', rmse_ensemble)
print('Out of sample rmse single:', rmse_single)

#%%
plt.rcParams['figure.dpi']= 100
sp = 1
fig = plt.figure(figsize=(5,5))
plt.plot(np.linspace(0,length_training-1,length_training-1), training_set[1:length_training,sp], color = 'b',
            label = 'Data')
plt.plot(np.linspace(0,length_training-1,length_training-1), train_fit[0][:,sp], color = 'k', 
            label = 'Minimum error')
plt.plot(np.linspace(0,length_training-1,length_training-1), train_ens[:,sp], color = 'red',
            label = 'Ensemble')
plt.fill_between(np.linspace(0,length_training-1,length_training-1), train_ens[:,sp]-1.96*train_err[:,sp],
                                                     train_ens[:,sp]+1.96*train_err[:,sp],
                                                     alpha = 0.5,color= 'r')

fig = plt.figure(figsize = (10,6))
for sp in range(np.shape(test_data)[1]):
   number='23'+str(sp+1)
   plt.subplot(number)
   titolo='Species '+str(sp+1)
   plt.title(titolo)
   plt.plot(np.linspace(0,orizzonte,orizzonte), test_data[:,sp], color = 'b',
            label = 'Data') 
   plt.plot(np.linspace(0,orizzonte,orizzonte), cv_forecast[:,sp], color = 'k', 
            label = 'Minimum error')
   plt.plot(np.linspace(0,orizzonte,orizzonte), pred[:,sp], color = 'red',
            label = 'Ensemble')
   plt.fill_between(np.linspace(0,orizzonte,orizzonte), pred[:,sp]-1.96*err[:,sp],
                                                     pred[:,sp]+1.96*err[:,sp],
                                                     alpha = 0.5,color= 'r')
                  

   for k in range(len(forecast)):
	   plt.plot(np.linspace(0,orizzonte,orizzonte), forecast[k][:,sp], color = 'g', alpha = 0.5)
   if sp ==0:
      plt.legend()


#### VCR inference
#%%
true_jacobian_matrix=fn.unfold_jacobian(analytical_jacobian,np.shape(ts)[1])
true_vcr = np.delete(preprocessing.scale(fn.vcr(true_jacobian_matrix)), 0)
ensemble_vcr = preprocessing.scale(fn.vcr(jac_ens))
vcr_err = fn.error_on_vcr(jac_err)
plt.rcParams['figure.dpi']= 100
fig = plt.figure()
plt.title('scaled volume contraction rate')
plt.plot(true_vcr, color = 'b', label = 'True')
plt.plot(cv_vcr , color = 'g', label = 'Minimum error')
plt.plot(ensemble_vcr , color = 'r', label = 'Ensemble')
plt.fill_between(np.linspace(0,len(ensemble_vcr)-1,len(ensemble_vcr)),
						 ensemble_vcr-1.96*np.array(vcr_err),
                                                 ensemble_vcr+1.96*np.array(vcr_err),
                                                 alpha = 0.5,color= 'r')
plt.legend()
print('VCR inference quality:\n',
'####\n',
'Correlation coefficient:\n',
'Cross Validation:', stat.pearsonr(true_vcr, cv_vcr)[0], '\n',
'Ensemble:', stat.pearsonr(true_vcr, ensemble_vcr)[0], '\n',
'####\n',
'RMSE:\n',
'Cross Validation:', fn.rmse(true_vcr, cv_vcr), '\n',
'Ensemble:', fn.rmse(true_vcr, ensemble_vcr))
print('Inference quality of the Jacobian matrix\n',
'####\n',
'Correlation coefficient:\n',
'Cross Validation:', np.nanmean(fn.inference_quality(jacobian_list[0], true_jacobian_matrix)), '\n',
'Ensemble:', np.nanmean(fn.inference_quality(jac_ens, true_jacobian_matrix)))


