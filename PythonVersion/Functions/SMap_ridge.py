import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import inv
from operator import itemgetter
import importlib
import functions as fn
importlib.reload(fn)

class SMRidge:
	'''
	Regularized S-map: Cenci et al., Methods Eco. Evol. (2019)
	Tikhonov regularization
	'''
	def __init__(self,l,t):
		self.l = l
		self.t = t
	def make_kernel(self, y, X):
		'''
		X are the predictors
		y is the predictee (1xd vector), i.e. x(t) before x(t+1)
		'''
		pairwise_dists = np.sqrt(np.sum((X-y)**2, axis = 1))
		K = np.exp(-self.t*pairwise_dists / np.mean(pairwise_dists))
		return(np.diag(K))
	def ridge_fit(self, X,Y,W, intercept = True):
		'''
		Solve for the parameters of a linear model 
		'''
		if intercept:
			X = np.column_stack((np.repeat(1,np.shape(X)[0]),X))
		else:
			X = np.column_stack((np.repeat(0,np.shape(X)[0]),X))
		J = inv(X.transpose().dot(W).dot(X) + self.l*np.identity(np.shape(X)[1])).dot(X.transpose()).dot(W).dot(Y)
		return(J[0], J[1:np.shape(X)[1]].transpose())
	def get_para(self,dat, intercept = True):
		'''
		Implement the fit here
		'''

		X,Y = fn.time_lagged_ts(dat)
		lib = range(np.shape(X)[0])
		pred = range(np.shape(X)[0])
		J = []
		c0 = []
		for ipred in range(len(lib)):
			libs = np.delete(lib,ipred)
			K = self.make_kernel(X[ipred,:],X[libs,:])
			c, jj = self.ridge_fit(X[libs,:], Y[libs,:],K,intercept)
			J.append(jj)
			c0.append(c)
		return(c0, J)

	def fit(self, X, c0, parameters):
		'''
		Implement the training set predictions here
		'''
		train_pred = []
		for tm in range(len(parameters)):
			train_pred.append(c0[tm] + np.dot(parameters[tm], X[tm,:].transpose()))
		return(np.stack(train_pred))

	def iterative_fit(self, M, intercept = True):
		#M,_ = fn.scale_training_data(M)
		Xx,Yy = fn.time_lagged_ts(M)
		Xx, _ = fn.scale_training_data(Xx)
		Yy, _ = fn.scale_training_data(Yy)
		K = self.make_kernel(Xx[(np.shape(Xx)[0]-1),:],Xx[0:(np.shape(Xx)[0]-1),:])
		c0, para = self.ridge_fit(Xx[0:(np.shape(Xx)[0]-1),:], Yy[0:(np.shape(Yy)[0]-1),:], K, intercept)
		return(c0, para)
	def predict(self, Xx, horizon, intercept = True):
		'''
		Make out-of-sample predictions
		X = training set
		horizon = length of prediction
		'''
		out_of_samp = []
		for n in range(horizon):
			c0, para = self.iterative_fit(Xx, intercept)
			prd = c0 +  np.dot(para, Xx[np.shape(Xx)[0]-1,:].transpose())
			Xx = np.vstack([Xx, prd])
			out_of_samp.append(prd)
		return(np.stack(out_of_samp))

	def score(self, X_true, X_predicted):
		'''
		Compute the RMSE
		'''
		return(np.sqrt(np.mean((X_true-X_predicted)**2)))


