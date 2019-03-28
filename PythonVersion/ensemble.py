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
length_training = 400
ts = np.loadtxt('input/deterministic_chaos_k.txt')
true_jacobian = np.loadtxt('input/jacobian_chaos_k.txt')
#ts = np.loadtxt('input/deterministic_chaos_lv.txt')
#true_jacobian = np.loadtxt('input/jacobian_chaos_lv.txt')
#ts = ts+ts*np.random.normal(0.,0.05*np.std(ts), size =np.shape(ts))
training_set = ts[0:length_training,:]
true_jacobian = true_jacobian[0:length_training,:]
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
#### In sample statistics
forecast = []
train_fit = []
vcr_list = []
orizzonte = 40
test_data = ts[length_training:(length_training+orizzonte),:]
rmse_ensemble_out = []
for n in range(len(l)):
    smap_object = smr.SMRidge(l[n],t[n])
    ### Training set
    c0, jacobians = smap_object.get_para(training_set)
    vcr_list.append(preprocessing.scale(fn.vcr(jacobians)))
    train_fit.append(smap_object.fit(training_set,c0, jacobians))
    ### Test set
    pred = smap_object.predict(training_set,orizzonte)
    forecast.append(fn.unscale_test_data(pred, scaler))

    ### This is just for plotting the landscape
    #scaled_test = scaler.transform(test_data)
    rmse_ensemble_out.append(smap_object.score(forecast[n],test_data))

lm = np.array([parameters[k]['lambda'] for k in range(len(full_error))])
th = np.array([parameters[k]['theta'] for k in range(len(full_error))])
a = np.column_stack([lm,th,full_error])
np.savetxt('output/training_landscape.txt',  a)
l = np.squeeze(np.asarray(l))
t = np.squeeze(np.asarray(t))
rmse_ensemble_out = np.squeeze(np.asarray(rmse_ensemble_out))
b = np.column_stack([l,t,rmse_ensemble_out])
np.savetxt('output/testing_landscape.txt',  b)
print(rmse_ensemble_out)

train_ens, train_err = fn.ensemble_forecast(train_fit,e)
pred, err = fn.ensemble_forecast(forecast,e)
infered_vcr, vcr_err  = fn.ensemble_vcr(vcr_list,e)
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
plt.rcParams['figure.dpi']= 100
sp = 0
fig = plt.figure(figsize=(5,5))
plt.plot(np.linspace(0,length_training-1,length_training-1), training_set[1:length_training,sp], color = 'b',
            label = 'Data')
plt.plot(np.linspace(0,length_training-1,length_training-1), train_fit[1][:,sp], color = 'k', 
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
for k in range(len(forecast)):
	plt.plot(np.linspace(0,orizzonte,orizzonte), forecast[k][:,sp], color = 'g', alpha = 0.5)

plt.legend()


#### VCR inference
#%%
true_jacobian=fn.unfold_jacobian(true_jacobian,np.shape(ts)[1])
true_vcr = np.delete(preprocessing.scale(fn.vcr(true_jacobian)), 0)

plt.rcParams['figure.dpi']= 100
fig = plt.figure()
plt.title('scaled volume contraction rate')
plt.plot(true_vcr, color = 'b')
plt.plot(vcr_list[0] , color = 'k')
plt.plot(infered_vcr , color = 'r')
plt.fill_between(np.linspace(0,len(infered_vcr)-1,len(infered_vcr)),
						 infered_vcr-1.96*np.array(vcr_err),
                                                 infered_vcr+1.96*np.array(vcr_err),
                                                 alpha = 0.5,color= 'r')

print('VCR inference quality:\n',
'####\n',
'Correlation coefficient:\n',
'Cross Validation:', stat.pearsonr(true_vcr, vcr_list[0])[0], '\n',
'Ensemble:', stat.pearsonr(true_vcr, infered_vcr)[0], '\n',
'####\n',
'RMSE:\n',
'Cross Validation:', np.sqrt(np.mean((true_vcr - vcr_list[0])**2)), '\n',
'Ensemble:', np.sqrt(np.mean((true_vcr - infered_vcr)**2)))

plt.show()

