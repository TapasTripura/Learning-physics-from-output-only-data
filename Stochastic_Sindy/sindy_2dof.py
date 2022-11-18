"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). 
-- 'A sparse Bayesian framework for discovering interpretable nonlinear stochastic
    dynamical systems with Gaussian white noise'.
    (in Arxiv: 'Learning governing physics from output only measurements')
   
*** This is the 'Stochastic SINDy' code for the Example 3: 2DOF system.

# If you use any part of this code, then please cite us.

@article{tripura2022learning,
  title={Learning governing physics from output only measurements},
  author={Tripura, Tapas and Chakraborty, Souvik},
  journal={arXiv preprint arXiv:2208.05609},
  year={2022}
}

# The citation can also be found in Elsevier, MSSP journal.
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

# %%
import numpy as np
from numpy import linalg as LA
import utils_library
import utils_response
import matplotlib.pyplot as plt
from spyder_kernels.utils.iofuncs import load_dictionary

# %%

# Response generation:
T, dt = 1, 0.001
t = np.arange(0, T+dt, dt)
np.random.seed(0)
x1, x2, x3, x4 = 0.5, 0, 0.05, 0 # initial displacement for 2dof
xdrift1, xdrift2, _, _, _, y1, y2, y3, y4, _ = utils_response.dof2sys(x1, x2, x3, x4, T)

# Expected Dictionary Creation:
libr = []
for j in range(len(y1)):
    data = np.row_stack((y1[j,0:-1], y2[j,0:-1], y3[j,0:-1], y4[j,0:-1]))
    Dtemp, nl = utils_library.library(data, 3, 0)
    libr.append(Dtemp)
libr = np.array(libr)
D1 = np.mean(libr, axis = 0)

# compute Sparse regression: sequential least squares
def sindy(lam,D,dxdt): 
    Xi = np.matmul(np.linalg.pinv(D), dxdt.T) # initial guess: Least-squares
    for k in range(50):
        smallinds = np.where(abs(Xi) < lam)   # find small coefficients
        Xi[smallinds] = 0
        for ind in range(Xi.shape[1]):
            biginds = np.where(abs(Xi[:,ind]) > lam)
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.matmul(np.linalg.pinv(D[:, biginds[0]]), dxdt[ind, :].T) 
    return Xi

# %%
"""
    Identification of Drifts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
dxdt1 = np.stack(( xdrift1, xdrift2 )) 

# Adding noise:
eps = 0.0      # noise strength
dxdt1 = dxdt1 + np.multiply( np.vstack(eps*np.std(dxdt1, axis=1)), \
                          np.random.normal(0, 1, (dxdt1.shape[0], dxdt1.shape[1])) )
    
lam = 0.5      # lam is our sparsification knob.
Xi_drift = sindy(lam,D1,dxdt1)

print(Xi_drift)

# %%
"""
    Identification of Diffusions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Response generation:
np.random.seed(0)
x1, x2, x3, x4 = 0.001, 0, 0.001, 0 # initial displacement for 2dof
_, _, xdiff11, _, xdiff22, y1, y2, y3, y4, _ = utils_response.dof2sys(x1, x2, x3, x4, T)

# Dictionary Creation:
libr = []
for j in range(len(y1)):
    data = np.row_stack((y1[j,0:-1], y2[j,0:-1], y3[j,0:-1], y4[j,0:-1]))
    Dtemp, nl = utils_library.library(data, 3, 0)
    libr.append(Dtemp)
libr = np.array(libr)
D2 = np.mean(libr, axis = 0)

# Adding noise:
dxdt2 = np.stack(( xdiff11, xdiff22 )) 

# Adding noise:
eps = 0.0      # noise strength
dxdt2 = dxdt2 + np.multiply( np.vstack(eps*np.std(dxdt2, axis=1)), \
                          np.random.normal(0, 1, (dxdt2.shape[0], dxdt2.shape[1])) )

lam = 90
Xi_diff = sindy(lam,D2,dxdt2)
Xi_diff[np.where(Xi_diff < 0)] = 0  # removing the negative values to avoid nan values,
Xi_diff[np.where(Xi_diff > 100)] = 0  # removing the higher values to avoid overflow,

print(Xi_diff)

# %%
def sparse_sol(xinit, params):
    T, xi_drift, xi_diff, polyorder, harmonic = params
    
    dt = 0.001
    t = np.arange(0, T+dt, dt)
    Nsamp = 200
    
    # np.random.seed(2021)
    y1, y2, y3, y4 = [], [], [], []
    for ensemble in range(Nsamp):
        x0 = np.zeros(4)
        x0[0] = xinit[0] 
        x0[2] = xinit[1]
        x = np.vstack(x0)

        for n in range(len(t)-1):
            dW = np.sqrt(dt)*np.random.randn(2)
            
            D, nl = utils_library.library(np.vstack(x0), polyorder, harmonic)
            a = np.dot(D, xi_drift).reshape(-1)
            b = np.dot(D, Xi_diff).reshape(-1)
            b = np.sqrt(b)
            
            sol1 = x0[0] + x0[1]*dt
            sol2 = x0[1] + a[0]*dt + b[0]*dW[0]
            sol3 = x0[2] + x0[3]*dt
            sol4 = x0[3] + a[1]*dt + b[1]*dW[1]
            x0 = np.array(( sol1.squeeze(), sol2.squeeze(), sol3.squeeze(), sol4.squeeze() ))
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
        y4.append(x[3,:])
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    
    # Get an index of columns which has any NaN value
    index1 = np.isnan(y1).any(axis=1)
    index2 = np.isnan(y2).any(axis=1)
    index3 = np.isnan(y3).any(axis=1)
    index4 = np.isnan(y4).any(axis=1)
    # Delete columns with any NaN value from 2D NumPy Array
    y1 = np.delete(y1, index1, axis=0)
    y2 = np.delete(y2, index2, axis=0)
    y3 = np.delete(y3, index3, axis=0)
    y4 = np.delete(y4, index4, axis=0)
    return np.array( [np.mean(y1, axis=0), np.mean(y2, axis=0), \
                    np.mean(y3, axis=0), np.mean(y4, axis=0)] )

# %%
""" The validation and predictions """

# Response generation:
T, dt = 4, 0.001
t = np.arange(0, T+dt, dt)
np.random.seed(0)
x1, x2, x3, x4 = 0.5, 0, 0.05, 0 # initial displacement for 2dof
_, _, _, _, _, y1, y2, y3, y4, _ = utils_response.dof2sys(x1, x2, x3, x4, T)

# %%
# integrate true and identified systems
np.random.seed(0)
params = [T, Xi_drift, Xi_diff, 3, 0]
xt_pred = sparse_sol(np.array([x1,x3]), params)

# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24
xt = np.array([np.mean(y1, axis=0), np.mean(y2, axis=0), np.mean(y3, axis=0), np.mean(y4, axis=0)])

fig1 = plt.figure(figsize = (12,6))
plt.subplot(2,2,1); plt.plot(t, xt[0,:], 'r', linewidth = 1.5)
plt.subplot(2,2,1); plt.plot(t, xt[1,:], 'k', linewidth = 1.2)
plt.subplot(2,2,1); plt.plot(t, xt_pred[0,:], 'b--', linewidth = 1.5)
plt.subplot(2,2,1); plt.plot(t, xt_pred[1,:], 'g--', linewidth = 1.2)
plt.subplot(2,2,1); plt.xlabel('Time (s)')
plt.subplot(2,2,1); plt.ylabel('States')

plt.subplot(2,2,3); plt.plot(t, xt[2,:], 'r', linewidth = 1.5)
plt.subplot(2,2,3); plt.plot(t, xt[3,:], 'k', linewidth = 1.2)
plt.subplot(2,2,3); plt.plot(t, xt_pred[2,:], 'b--', linewidth = 1.5)
plt.subplot(2,2,3); plt.plot(t, xt_pred[3,:], 'g--', linewidth = 1.2)
plt.subplot(2,2,1); plt.xlabel('Time (s)')
plt.subplot(2,2,1); plt.ylabel('States')

plt.subplot(2,2,2); plt.plot(xt[0,:], xt[1,:], 'r')
plt.subplot(2,2,2); plt.plot(xt_pred[0,:], xt_pred[1,:], 'b--')
plt.subplot(2,2,2); plt.xlabel('$X_1$')
plt.subplot(2,2,2); plt.ylabel('$X_2$')

plt.subplot(2,2,2); plt.plot(xt[0,:], xt[1,:], 'r')
plt.subplot(2,2,2); plt.plot(xt_pred[0,:], xt_pred[1,:], 'b--')
plt.subplot(2,2,2); plt.xlabel('$X_1$')
plt.subplot(2,2,2); plt.ylabel('$X_2$')

plt.subplot(2,2,4); plt.plot(xt[2,:], xt[3,:], 'r')
plt.subplot(2,2,4); plt.plot(xt_pred[2,:], xt_pred[3,:], 'b--')
plt.subplot(2,2,4); plt.xlabel('$X_1$')
plt.subplot(2,2,4); plt.ylabel('$X_2$')
