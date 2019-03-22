import numpy as np
import SMap_ridge as smr
import functions as fn
from operator import itemgetter
from joblib import Parallel, delayed
import multiprocessing


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

def fun_rolling(i, grid, dat, orizzonte):
	'''
	This is for convenience so that if you want to run 
	any rolling cross validation in parallel you can
	do it easily
	'''
	iterations = 30
	r = smr.SMRidge(grid[i]['lambda'], grid[i]['theta'])
	val_err = []
	for n in range(iterations):
		tr_dat = dat[0:(np.shape(dat)[0]-iterations+n-orizzonte),:]
		tr_dat, scaler_cv = fn.scale_training_data(tr_dat)
		prd = r.predict(tr_dat,orizzonte)
		prd = fn.unscale_test_data(prd, scaler_cv)
		val_dat = dat[(np.shape(dat)[0]-iterations+n-orizzonte):(np.shape(dat)[0]-iterations+n),:]
		val_err.append(r.score(val_dat,prd))
	return(np.mean(val_err))

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
	error = [fun_rolling(i, grid,dat, orizzonte) for i in range(len(grid))]
	idx = min(enumerate(error), key=itemgetter(1))[0]
	return(error, grid[idx]['lambda'], grid[idx]['theta'])



def ensemble_rollingcv(grid,dat, orizzonte, par = False):
	'''
	Same as before 
	but now we take the validation error of the best models over the iterations

	Important note: at the moment the non-parallel version is faster than the parallel counterpart
	'''
	if par:
		num_cores = multiprocessing.cpu_count() - 1
		error = Parallel(n_jobs=num_cores, backend='threading')(delayed(fun_ensemble)(i, grid,dat, orizzonte) for i in range(len(grid)))
	else:
		print('Running not in parallel: safest choice at the moment')
		error = [fun_ensemble(i, grid,dat, orizzonte) for i in range(len(grid))]
	lam = []
	tht = []
	er = []
	min_error = min(error)
	max_error = max(error)
	prog = True
	while prog:
		idx,val = min(enumerate(error), key=itemgetter(1))
		lam.append(np.round(grid[idx]['lambda'],5))
		tht.append(np.round(grid[idx]['theta'],5))
		er.append(np.round(error[idx],4))
		## Set the current minimum value to the max
		## so that the second smallest value is the new minimum
		error[idx] = max_error
		if val>min_error*1.2:
			prog = False
	return(er,lam,tht)