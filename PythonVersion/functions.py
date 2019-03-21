import numpy as np
from sklearn import preprocessing
import scipy.stats as stat

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

def vcr(X):
	vol_contraction = [np.trace(X[n]) for n in range(len(X))]
	return(vol_contraction)

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
			true_ij = [Y[n][i,j] for n in range(len(X))]
			M[i,j] = stat.pearsonr(inf_ij, true_ij)[0]
	return(M)
	