import random, heapq
import pylab as plt
from scipy import integrate
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import scipy
import os, sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib2tikz import save as tikz_save
import numdifftools as nd

r = np.array([1., 0.72, 1.53, 1.27])
A = np.matrix([[1.*r[0], 1.09*r[0], 1.52*r[0], 0.*r[0]],  [0.*r[1], 1.*r[1], 0.44*r[1],1.36*r[1]],  [2.33*r[2], 0.*r[2], 1.*r[2], 0.47*r[2]], [1.21*r[3], 0.51*r[3], 0.35*r[3], 1.*r[3]]])
###### Initial conditions and time steps #######
x0 = 0.2; y0 = 0.2; z0 = 0.3; k0 = 0.3;
####### At 1015 the model explode
T = 5000;
dt = 0.01;
n_steps = T/dt;
t = np.linspace(0, T, n_steps)
X_f1 = np.array([x0, y0, z0, k0])
################################################
def dX_dt(X, t = 0):
    dydt = np.array([X[s]*(r[s] - np.sum(np.dot(A,X)[0,s]))for s in range(0,len(X))])
    return(dydt)

ini_cond = integrate.odeint(dX_dt, X_f1, t)
X_f1 = np.array([ini_cond[len(t)-1,0], ini_cond[len(t)-1,1], ini_cond[len(t)-1,2], ini_cond[len(t)-1,3]])
ts = integrate.odeint(dX_dt, X_f1, t)

g = open('../inputFiles/deterministic_chaos_lv.txt', 'w')
jacobian_matrix = open('../inputFiles/jacobian_chaos_lv.txt', 'w')
num_species = ts.shape[1]
for i in range(0,ts.shape[0]):
	if i%200 == 0:
		f_jacob = nd.Jacobian(dX_dt)(np.squeeze(np.asarray(ts[i,:])))
		g.write('%f %f %f %lf\n' % (ts[i, 0], ts[i, 1], ts[i, 2],ts[i,3]))
		for u in range(0,num_species):
			for z in range(0,num_species):
			    	jacobian_matrix.write('%lf ' % (f_jacob[u,z]))
		jacobian_matrix.write('\n')
g.close()
jacobian_matrix.close()
