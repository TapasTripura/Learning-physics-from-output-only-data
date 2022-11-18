"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). 
-- 'A sparse Bayesian framework for discovering interpretable nonlinear stochastic
    dynamical systems with Gaussian white noise'.
    (in Arxiv: 'Learning governing physics from output only measurements')
   
*** This is the 'Stochastic SINDy' code for the Example 2: Parametrically excited Duffing oscillator.

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
import utils_library
import utils_response
import matplotlib.pyplot as plt
from spyder_kernels.utils.iofuncs import load_dictionary

# %%

# Response generation:
T, dt = 1, 0.001
t = np.arange(0, T+dt, dt)
np.random.seed(0)
x1, x2 = 0.05, 0 # initial displacement for duffing
xdt, bxt, y1, y2, t_eval = utils_response.duffing(x1, x2, T)

# Expected Dictionary Creation:
libr = []
for j in range(len(y1)):
    data = np.row_stack((y1[j,0:-1], y2[j,0:-1]))
    Dtemp, nl = utils_library.library(data, 6, 0)
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
For the drift identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
np.random.seed(11)
# Adding noise:
eps = 0.01      # noise strength
dxdt = np.column_stack(( xdt + np.random.normal(0, eps*np.std(xdt), len(xdt)) ))

lam = 1000 # 1000 # 0.5      # lam is our sparsification knob.
Xi_drift = sindy(lam,D1,dxdt)

theta1 = np.zeros(36)
theta1[[1,2,6]] = -1000,-2,-100000
error = np.sum((theta1 - Xi_drift)**2)/np.sum(theta1**2)/100
print(error)

print(Xi_drift)

# %%
"""
For the diffusion identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Response generation:
np.random.seed(0)
x1, x2 = 0.05, 0 # initial displacement for duffing
xdt, bxt, y1, y2, t_eval = utils_response.duffing(x1, x2, T)

# Expected Dictionary Creation:
libr = []
for j in range(len(y1)):
    data = np.row_stack((y1[j,0:-1], y2[j,0:-1]))
    Dtemp, nl = utils_library.library(data, 6, 0)
    libr.append(Dtemp)
libr = np.array(libr)
D2 = np.mean(libr, axis = 0)

# %%
# Adding noise:
np.random.seed(11)
eps = 0.01
xdts = np.column_stack(( bxt + np.random.normal(0, eps*np.std(bxt), len(bxt)) ))

#
lam = 0.5 # 1000 # 0.5
Xi_diff = sindy(lam,D2,xdts)
Xi_diff[3] = Xi_diff[3] - 1000
Xi_diff[np.where(Xi_diff < 0)] = 0  # removing the negative values to avoid nan values,
Xi_diff[np.where(Xi_diff > 10**5)] = 0  # removing the higher values to avoid overflow,

# 
theta2 = np.zeros(36)
theta2[3] = 10
error2 = np.sum((theta2 - np.sqrt(Xi_diff))**2)/np.sum(theta2**2)/100
print(error2)

print(Xi_diff)

# %%
def sparse_sol(xinit, params):
    T, xi_drift, xi_diff, polyorder, harmonic = params
    
    dt = 0.001
    t = np.arange(0, T+dt, dt)
    Nsamp = 200
    
    # np.random.seed(2021)
    y1, y2 = [], []
    for ensemble in range(Nsamp):
        x0 = np.zeros(2)
        x0[0] = xinit 
        x = np.vstack(x0)
        for n in range(len(t)-1):
            dW = np.sqrt(dt)*np.random.randn()
            
            D, nl = utils_library.library(np.vstack(x0), polyorder, harmonic)
            a = np.dot(D, xi_drift).reshape(-1)
            b = np.dot(D, Xi_diff).reshape(-1)
            b = np.sqrt(b)
            
            sol1 = x0[0] + x0[1]*dt
            sol2 = x0[1] + a*dt + b*dW
            x0 = np.array((sol1.squeeze(), sol2.squeeze()))
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    # Get an index of columns which has any NaN value
    index1 = np.isnan(y1).any(axis=1)
    index2 = np.isnan(y2).any(axis=1)
    # Delete columns with any NaN value from 2D NumPy Array
    y1 = np.delete(y1, index1, axis=0)
    y2 = np.delete(y2, index2, axis=0)
    return np.array([np.mean(y1, axis=0), np.mean(y2, axis=0)])

# %%
""" The validation and predictions """

# Response generation:
T, dt = 10, 0.001
t = np.arange(0, T+dt, dt)
x1, x2 = 0.05, 0 # initial displacement for duffing
np.random.seed(0)
_, _, y1, y2, _ = utils_response.duffing(x1, x2, T)

# %%
# integrate true and identified systems
np.random.seed(0)
params = [T, Xi_drift, Xi_diff, 6, 0]
xt_pred = sparse_sol(x1, params)

# %%
data_dict = load_dictionary('Revision_DVP_diff10_final.spydata')
xt = data_dict[0]['xt']
xt_ii = data_dict[0]['xt_ii']
xt_l = data_dict[0]['xt_l']
xt_u = data_dict[0]['xt_u']

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 26

fig11 = plt.figure(figsize=(10,6))
plt.plot(t[:5001], xt[0,:], color='r', label='True')
plt.plot(t[:5001], xt_ii[0,:], ':', color='k', linewidth=1.5, label='Identified')
plt.plot(t[:5001], xt_pred[0,:5001], 'b-.', label='SINDy')
plt.fill_between(t[:5001], xt_u[0,:], xt_l[0,:], alpha = .25, color = 'c')
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel(' $X$(t) ')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.legend(loc=1, ncol=2, columnspacing=0.7, handlelength=0.9, handletextpad=0.25)
plt.xlim([0,4])
plt.margins(0)
tp1, tp2 = 2000, 2201
# location for the zoomed portion 
sub_axes1 = plt.axes([.62, .2, .25, .2]) 
sub_axes1.plot(t[tp1:tp2], xt[0,tp1:tp2], color='r')
sub_axes1.plot(t[tp1:tp2], xt_ii[0,tp1:tp2], ':', c = 'k')
sub_axes1.plot(t[tp1:tp2], xt_pred[0,tp1:tp2], 'b-.')
sub_axes1.fill_between(t[tp1:tp2], xt_l[0,tp1:tp2], xt_u[0,tp1:tp2], alpha = .25, color = 'c')
sub_axes1.margins(0)

# %%
fig21 = plt.figure(figsize=(10,6))
plt.plot(t[:5001], xt[1,:], color='r', label='True')
plt.plot(t[:5001], xt_ii[1,:], ':', color='k', linewidth=1.5, label='Identified')
plt.plot(t[:5001], xt_pred[1,:5001], 'b-.', label='SINDy')
plt.fill_between(t[:5001], xt_u[1,:], xt_l[1,:], alpha = .25, color = 'c')
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel(' $X$(t) ')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.legend(loc=1, ncol=2, columnspacing=0.7, handlelength=0.9, handletextpad=0.25)
plt.xlim([0,4])
plt.margins(0)
tp1, tp2 = 2000, 2201
# location for the zoomed portion 
sub_axes1 = plt.axes([.62, .2, .25, .2]) 
sub_axes1.plot(t[tp1:tp2], xt[1,tp1:tp2], color='r')
sub_axes1.plot(t[tp1:tp2], xt_ii[1,tp1:tp2], ':', c = 'k')
sub_axes1.plot(t[tp1:tp2], xt_pred[1,tp1:tp2], 'b-.')
sub_axes1.fill_between(t[tp1:tp2], xt_l[1,tp1:tp2], xt_u[1,tp1:tp2], alpha = .25, color = 'c')
sub_axes1.margins(0)

# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24
xt = np.array([np.mean(y1, axis=0), np.mean(y2, axis=0)])

fig1 = plt.figure(figsize = (12,6))
plt.subplot(1,2,1); plt.plot(t, xt[0,:], 'r', linewidth = 1.5)
plt.subplot(1,2,1); plt.plot(t, xt[1,:], 'k', linewidth = 1.2)

plt.subplot(1,2,1); plt.plot(t, xt_pred[0,:], 'b--', linewidth = 1.5)
plt.subplot(1,2,1); plt.plot(t, xt_pred[1,:], 'g--', linewidth = 1.2)

plt.subplot(1,2,1); plt.xlabel('Time (s)')
plt.subplot(1,2,1); plt.ylabel('States')

plt.subplot(1,2,2); plt.plot(xt[0,:], xt[1,:], 'r')
plt.subplot(1,2,2); plt.plot(xt_pred[0,:], xt_pred[1,:], 'b--')

plt.subplot(1,2,2); plt.xlabel('$X_1$')
plt.subplot(1,2,2); plt.ylabel('$X_2$')
