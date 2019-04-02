import pylab as plt
from scipy import integrate
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import scipy
import numdifftools as nd
import sys, random
a = 5.; b = 3.5;
###### Initial conditions and time steps #######
x0 = .1; y0 = .1; z0 = .2;
T = 500.;
dt = 0.001;
n_steps = T/dt;
t = np.linspace(0, T, n_steps)
X_f1 = np.array([x0, y0, z0])
######## Auxiliar Functions ####################
def dX_dt(X, t = 0):
	return(np.array([X[2],
			 -X[2]*(a*X[1] + b*X[1]**2 + X[0]*X[2]),
			X[0]**2 - abs(X[0])*X[1] + X[1]**2 -1]))

ts = integrate.odeint(dX_dt, X_f1, t)
X_f1 = ts[ts.shape[0]-1,:]
ts = integrate.odeint(dX_dt, X_f1, t)
g = open('../inputFiles/deterministic_chaos_hs.txt', 'w')
jacobian_matrix = open('../inputFiles/jacobian_chaos_hs.txt', 'w')
num_species = ts.shape[1]
for i in range(0,ts.shape[0]):
	if i%100 == 0:
		f_jacob = nd.Jacobian(dX_dt)(np.squeeze(np.asarray(ts[i,:])))
		g.write('%f %f %f\n' % (ts[i, 0], ts[i, 1], ts[i, 2]))
		for u in range(0,num_species):
			for z in range(0,num_species):
			    	jacobian_matrix.write('%lf ' % (f_jacob[u,z]))
		jacobian_matrix.write('\n')
g.close()
jacobian_matrix.close()
