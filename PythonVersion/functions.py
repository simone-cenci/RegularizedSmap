import numpy as np
from sklearn import preprocessing
import scipy.stats as stat
from statsmodels.stats.weightstats import DescrStatsW
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
def ensemble_vcr(X,E):
	dim0 = np.shape(X[0])[0]
	M = [0]*dim0
	S = [0]*dim0
	w = make_weights(E)
	for i in range(dim0):
		weighted_stats = DescrStatsW([X[n][i] for n in range(len(X))], w, ddof=0)
		M[i] = weighted_stats.mean
		S[i] = weighted_stats.std/np.sqrt(len(X))
	return(M,S)

def put_j_together(X):
	J=[]
	dim = np.shape(X[0][0])[0]
	for n in range(len(X[0])):
		#J.append(np.reshape(np.stack([X[0][n],X[1][n],X[2][n],X[3][n],X[4][n]]), (5,5)).transpose())
		J.append(np.reshape(np.stack([X[k][n] for k in range(dim)]), (dim,dim)).transpose())
	return(J)
