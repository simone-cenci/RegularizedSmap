import numpy as np
from sklearn import preprocessing
import scipy.stats as stat
from statsmodels.stats.weightstats import DescrStatsW
import SMap_ridge as smr


def time_lagged_ts(dtset, specie = 'all', look_back=1):
	'''
	Input dtset = time series
	look_back = time lage of predictions
	Output:
	dataX = predictors
	dataY = predictee (time lagged)
	'''
	if specie == 'all':
	        dataX = np.zeros((np.shape(dtset)[0] - look_back, np.shape(dtset)[1]))
	        dataY = np.zeros((np.shape(dtset)[0] - look_back, np.shape(dtset)[1]))
	        for i in range(np.shape(dtset)[0] - look_back):
	                dataX[i,:] = dtset[i:(i+look_back), :]
	                dataY[i,:] = dtset[i+look_back,:]
	else:
	        dataX = np.zeros((np.shape(dtset)[0] - look_back, np.shape(dtset)[1]))
	        dataY = np.zeros((np.shape(dtset)[0] - look_back, 1))
	        for i in range(np.shape(dtset)[0] - look_back):
	                dataX[i,:] = dtset[i:(i+look_back), :]
	                dataY[i,0] = dtset[i+look_back,specie]
	return np.array(dataX), np.array(dataY)

def scale_training_data(ts_training):
	'''
	This function scale (zero mean, unitary variance) the training data and return
	Both the scaled data and the scaling parameters to use to scale back 
	'''
	scaler_ts_training = preprocessing.StandardScaler().fit(ts_training)
	ts_training = preprocessing.scale(ts_training)	
	return(ts_training, scaler_ts_training)

def unscale_test_data(predicted_data, scaler_ts_training):
	'''
	This function scale back the predicted data to the original scale so to compare with training set
	'''
	pred = scaler_ts_training.inverse_transform(predicted_data)
	return(pred)
def unfold_jacobian(X,dim):
	j = []

	for n in range(np.shape(X)[0]):
		j.append(X[n,:].reshape(dim,dim))
	return(j)
def vcr(X):
	vol_contraction = [np.trace(X[n]) for n in range(len(X))]
	return(vol_contraction)

def error_on_vcr(X):
	'''
	The error on the volume contraction rate is sqrt(sum(deltaJ_ii^2))
	'''
	eps_vcr = [np.sqrt(np.sum(np.diag(X[n])**2)) for n in range(len(X))]
	return(eps_vcr)

def rmse(X,Y):
	return(np.sqrt(np.mean((X-Y)**2)))

def inference_quality(X,Y):
	'''
	Compute the correlation coefficient between infered and true Jacobians
	X = infered jacobians
	Y = true jacobians
	'''
	dim = np.shape(X[0])[0]
	M = np.zeros(shape=(dim,dim))
	for i in range(dim):
		for j in range(dim):
			inf_ij = [X[n][i,j] for n in range(len(X))]
			true_ij = [Y[n+1][i,j] for n in range(len(X))]
			M[i,j] = stat.pearsonr(inf_ij, true_ij)[0]
	return(M)

### Some function for the ensemble method
def make_weights(E):
	Z = np.sum([1./(n+1)*np.exp(-E[n]) for n in range(len(E))])
	w = [1./(n+1)*np.exp(-E[n])/Z for n in range(len(E))]
	return(w)

def ensemble_forecast(X,E):
	'''
	Compute the forecast from the ensemble
	'''
	dim0 = np.shape(X[0])[0]
	dim1 = np.shape(X[0])[1]
	M = np.zeros(shape=(dim0,dim1))
	S = np.zeros(shape=(dim0,dim1))
	w = make_weights(E)
	for i in range(dim0):
		for j in range(dim1):
			weighted_stats = DescrStatsW([X[n][i,j] for n in range(len(X))], w, ddof=0)
			M[i,j] = weighted_stats.mean
			S[i,j] = weighted_stats.std/np.sqrt(len(X))
	return(M,S)

def ensemble_jacobians(X,E):
	'''
	Compute the time series of Jacobian coefficients from the ensemble
	'''
	dimEnse = np.shape(X)[0]
	dimSeries = np.shape(X)[1]
	dim = np.shape(X)[2]

	M = np.zeros(shape=(dimSeries,dim,dim))
	S = np.zeros(shape=(dimSeries,dim,dim))
	w = make_weights(E)
	for s in range(dimSeries):
		for i in range(dim):
			for j in range(dim):
				weighted_stats = DescrStatsW([X[n][s][i,j] for n in range(dimEnse)], w, ddof=0)
				M[s][i,j] = weighted_stats.mean
				S[s][i,j] = weighted_stats.std/np.sqrt(dimEnse)
	return(M,S)
def ensemble_method(train_set, lmb, tht, h, scaler_, eps):
	'''
	Get in-sample and out of sample statistics for the ensemble method:
	1) For each lambda and theta in the ensemble compute prediction and inference
	2) Take a weighted mean with weights inversely proportional to the training error
	lmb,tht = lambda, theta
	h = orizzonte
	'''
	forecast = []
	train_fit = []
	jacobian_list = []
	for n in range(len(lmb)):
		smap_object = smr.SMRidge(lmb[n],tht[n])
		### Training set
		c0, jacobians = smap_object.get_para(train_set)
		jacobian_list.append(jacobians)
		train_fit.append(smap_object.fit(train_set,c0, jacobians))
		### Test set
		pred = smap_object.predict(train_set,h)
		forecast.append(unscale_test_data(pred, scaler_))

	train_ens, train_err = ensemble_forecast(train_fit,eps)
	pred, err = ensemble_forecast(forecast,eps)
	jac_ens, jac_err  = ensemble_jacobians(jacobian_list,eps)
	cv_forecast = forecast[0]
	cv_vcr = preprocessing.scale(vcr(jacobian_list[0]))
	return(train_fit, train_ens, train_err, forecast, cv_forecast, \
		pred, err, cv_vcr, jacobian_list, jac_ens, jac_err)

