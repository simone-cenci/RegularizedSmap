import numpy as np
import numdifftools as nd

g = open('Deterministic_TimeSeries.txt', 'r')
time_series = np.asmatrix([map(float,line.split(' ')) for line in g ])
time_series = time_series.astype('float32')
### Matrix and growth rates
r = np.array([1., 0.72, 1.53, 1.27])
A = np.matrix([[1.*r[0], 1.09*r[0], 1.52*r[0], 0.*r[0]],  [0.*r[1], 1.*r[1], 0.44*r[1],1.36*r[1]],  [2.33*r[2], 0.*r[2], 1.*r[2], 0.47*r[2]], [1.21*r[3], 0.51*r[3], 0.35*r[3], 1.*r[3]]])
def dX_dt(X, t = 0):
    dydt = np.array([X[s]*(r[s] - np.sum(np.dot(A,X)[0,s]))for s in range(0,len(X))])
    return(dydt)
### Compute the analytical Jacobian
num_species = time_series.shape[1]
jacobian_matrix = open('jacobian_chaos.txt', 'w')
for k in range(0,time_series.shape[0]):
	point = np.squeeze(np.asarray(time_series[k,:]))
	f_jacob = nd.Jacobian(dX_dt)(point)
	for u in range(0,num_species):
		for z in range(0,num_species):
		    	jacobian_matrix.write('%lf ' % (f_jacob[u,z]))
	jacobian_matrix.write('\n')
jacobian_matrix.close()
