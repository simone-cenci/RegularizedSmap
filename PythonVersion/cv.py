import numpy as np
import SMap_ridge as smr
import functions as fn
from operator import itemgetter
import importlib
importlib.reload(smr)
importlib.reload(fn)

def loocv(grid, dat):
	'''
	Leave one out cross validation (automatically implemented in the fit)
	'''
	error = []
	for k in range(len(grid)):
		r = smr.SMRidge(grid[k]['lambda'], grid[k]['theta'])
		j = r.get_para(dat)
		y_pred = r.fit(dat,j)
		error.append(r.score(dat[1:(np.shape(dat)[0]),:], y_pred))
	idx = min(enumerate(error), key=itemgetter(1))[0]
	return(error, grid[idx]['lambda'], grid[idx]['theta'])

def rollingcv(grid,dat, orizzonte):
	'''
	Here I run rolling window cross validation. That is:
	calling ---- the training data and **** the validation data
	we iteratively run:
	iter 1.) -------****
	iter 2.) --------****
	iter 3.) ---------****
	iter 4.) ----------****
	Then we take the validation error over the iterations
	'''
	error = []
	iterations = 30
	for k in range(len(grid)):
		r = smr.SMRidge(grid[k]['lambda'], grid[k]['theta'])
		val_err = []
		for n in range(iterations):
			tr_dat = dat[0:(np.shape(dat)[0]-iterations+n-orizzonte),:]
			tr_dat, scaler_cv = fn.scale_training_data(tr_dat)
			prd = r.predict(tr_dat,orizzonte)
			prd = fn.unscale_test_data(prd, scaler_cv)

			val_dat = dat[(np.shape(dat)[0]-iterations+n-orizzonte):(np.shape(dat)[0]-iterations+n),:]
			val_err.append(r.score(val_dat,prd))

		error.append(np.mean(val_err))
	idx = min(enumerate(error), key=itemgetter(1))[0]
	return(error, grid[idx]['lambda'], grid[idx]['theta'])
