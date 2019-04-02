import numpy as np
import SMap_ridge as smr
import functions as fn
from operator import itemgetter
from joblib import Parallel, delayed
import multiprocessing


import importlib
importlib.reload(smr)
importlib.reload(fn)


def fun_loocv(i,grid, dat, intercept = True):
	'''
	This is for convenience so that if you want to run
	any leave one out cross validation in parallel you can
	do it easily
	'''
	r = smr.SMRidge(grid[i]['lambda'], grid[i]['theta'])
	c, j = r.get_para(dat, intercept)
	y_pred = r.fit(dat, c, j)
	return(r.score(dat[1:(np.shape(dat)[0]),:], y_pred))
def fun_rolling(i, grid, dat, orizzonte, intercept = True):
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
		prd = r.predict(tr_dat,orizzonte, intercept)
		prd = fn.unscale_test_data(prd, scaler_cv)
		val_dat = dat[(np.shape(dat)[0]-iterations+n-orizzonte):(np.shape(dat)[0]-iterations+n),:]
		val_err.append(r.score(val_dat,prd))
	return(np.mean(val_err))

def loocv(grid, dat, par = False, intercept = True, ensemble = False):
	'''
	Leave one out cross validation (automatically implemented in the fit)
	'''
	if par:
		num_cores = multiprocessing.cpu_count() - 1
		print('Running in parallel')
		error = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(fun_loocv)(i, grid,dat, intercept) for i in range(len(grid)))
	else:
		print('Running not in parallel: safest choice at the moment')
		error = [fun_loocv(i, grid,dat, intercept) for i in range(len(grid))]
	
	### Ensemble method or note
	if ensemble:
		lam = []
		tht = []
		er = []
		full_error_path = error.copy()
		min_error = min(error)
		max_error = max(error)
		prog = True
		model=0
		max_models=100
		while prog:
			idx,val = min(enumerate(error), key=itemgetter(1))
			lam.append(np.round(grid[idx]['lambda'],5))
			tht.append(np.round(grid[idx]['theta'],5))
			er.append(np.round(error[idx],4))
			## Set the current minimum value to the max
			## so that the second smallest value is the new minimum
			error[idx] = max_error
			model+=1
			if val>min_error*1.2 or model==max_models:
				prog = False
		return(full_error_path, er,lam,tht)
	else:
		idx = min(enumerate(error), key=itemgetter(1))[0]
		return(error, grid[idx]['lambda'], grid[idx]['theta'])



def rollingcv(grid,dat, orizzonte, par = False, intercept = True, ensemble = False):
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
	if par:
		num_cores = multiprocessing.cpu_count() - 1
		print('Running in parallel')
		error = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(fun_rolling)(i, grid,dat, orizzonte, intercept) for i in range(len(grid)))
	else:
		print('Running not in parallel: safest choice at the moment')
		error = [fun_rolling(i, grid,dat, orizzonte, intercept) for i in range(len(grid))]
	if ensemble:
		lam = []
		tht = []
		er = []
		full_error_path = error.copy()
		min_error = min(error)
		max_error = max(error)
		prog = True
		model=0
		max_models=100
		while prog:
			idx,val = min(enumerate(error), key=itemgetter(1))
			lam.append(np.round(grid[idx]['lambda'],5))
			tht.append(np.round(grid[idx]['theta'],5))
			er.append(np.round(error[idx],4))
			## Set the current minimum value to the max
			## so that the second smallest value is the new minimum
			error[idx] = max_error
			model+=1
			if val>min_error*1.2 or model==max_models:
				prog = False
		return(full_error_path, er,lam,tht)
	else:
		idx = min(enumerate(error), key=itemgetter(1))[0]
		return(error, grid[idx]['lambda'], grid[idx]['theta'])

