from scipy import integrate
import numpy as np
import scipy
import numdifftools as nd

###### Define the parameters
def make_cr(every):
    nu1 = 0.1; nu2 = 0.07;
    C1 = 0.5; C2 = 0.5;
    lambda1 = 3.2; lambda2 =2.9;
    mu1 = 0.15; mu2 = 0.15;
    kappa1 = 2.5; kappa2 = 2.;
    Rstar = 0.3; k = 1.2;
    ###### Initial conditions and time steps #######
    p1_0 = 0.006884225; p2_0 = 0.087265965; c1_0 = 0.002226393; c2_0 = 1.815199890; r_0 = 0.562017616;
    T = 5000.;
    dt = 0.01;
    n_steps = T/dt;
    t = np.linspace(0, T, n_steps)
    X_f1 = np.array([p1_0, p2_0, c1_0, c2_0, r_0])
    ######## Model
    def Uptake(var_x, L, KI):
        return(L*var_x/(KI + var_x))
    def dX_dt(X, t = 0):
        dydt = np.array([nu1*Uptake(X[2], lambda1, C1)*X[0] - nu1*X[0],
                        nu2*Uptake(X[3], lambda2, C2)*X[1] - nu2*X[1],
                        mu1*Uptake(X[4], kappa1, Rstar)*X[2] - mu1*X[2] - nu1*Uptake(X[2], lambda1, C1)*X[0],
                        mu2*Uptake(X[4], kappa2, Rstar)*X[3] - mu2*X[3] - nu2*Uptake(X[3], lambda2, C2)*X[1],
                        X[4]*(1 - X[4]/k) -  mu1*Uptake(X[4], kappa1, Rstar)*X[2] - mu2*Uptake(X[4], kappa2, Rstar)*X[3]])
        return(dydt)
    ################
    ts = integrate.odeint(dX_dt, X_f1, t)
    jacobiano = []
    dat = []
    for i in range(0,ts.shape[0]):
        if i%every==0:
            f_jacob = nd.Jacobian(dX_dt)(np.squeeze(np.asarray(ts[i,:])))
            jacobiano.append(f_jacob)
            dat.append(ts[i,:])

    return(np.stack(dat), jacobiano)

def make_lv(every):
    ########## Parameters adapted from "Chaos in low-dimensional ...."
    r = np.array([1., 0.72, 1.53, 1.27])
    A = np.matrix([[1.*r[0], 1.09*r[0], 1.52*r[0], 0.*r[0]],  [0.*r[1], 1.*r[1], 0.44*r[1],1.36*r[1]],  [2.33*r[2], 0.*r[2], 1.*r[2], 0.47*r[2]], [1.21*r[3], 0.51*r[3], 0.35*r[3], 1.*r[3]]])


    ###### Initial conditions and time steps #######
    x0 = 0.2; y0 = 0.2; z0 = 0.3; k0 = 0.3;
    ####### At 1015 the model explode
    T = 1000;
    dt = 0.01;
    n_steps = T/dt;
    t = np.linspace(0, T, n_steps)
    X_f1 = np.array([x0, y0, z0, k0])
    ################################################
    ################################################
    def dX_dt(X, t = 0):
        dydt = np.array([X[s]*(r[s] - np.sum(np.dot(A,X)[0,s]))for s in range(0,len(X))])
        return(dydt)
    ################################################
    ts = integrate.odeint(dX_dt, X_f1, t)
    X_f1 = ts[ts.shape[0]-1,:]
    ts = integrate.odeint(dX_dt, X_f1, t)
    jacobiano = []
    dat = []
    for i in range(0,ts.shape[0]):
        if i%every==0:
            f_jacob = nd.Jacobian(dX_dt)(np.squeeze(np.asarray(ts[i,:])))
            jacobiano.append(f_jacob)
            dat.append(ts[i,:])

    return(np.stack(dat), jacobiano)

def make_hs(every):
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
    jacobiano = []
    dat = []
    for i in range(0,ts.shape[0]):
        if i%100 == 0:
            f_jacob = nd.Jacobian(dX_dt)(np.squeeze(np.asarray(ts[i,:])))
            jacobiano.append(f_jacob)
            dat.append(ts[i,:])
    return(np.stack(dat), jacobiano)        
