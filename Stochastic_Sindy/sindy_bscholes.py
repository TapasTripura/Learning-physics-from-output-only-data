"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). 
-- 'A sparse Bayesian framework for discovering interpretable nonlinear stochastic
    dynamical systems with Gaussian white noise'.
    (in Arxiv: 'Learning governing physics from output only measurements')
   
*** This is the 'Stochastic SINDy' code for the Example 1: Black-Scholes equation.

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

# %%
# Generate Data
np.random.seed(0)
T, x0 = 1, 1 # initial displacement for Bouc-Wen
dt = 0.001
t = np.arange(0, T+dt, dt)
xdt, xbt, y, t_eval = utils_response.blackscholes(x0, T)

# Expected Dictionary Creation:
libr = []
for j in range(len(y)):
    data = np.column_stack((y[j,0:-1]))
    Dtemp, nl = utils_library.library(data, 6, 0)
    libr.append(Dtemp)
libr = np.array(libr)
D1 = np.mean(libr, axis = 0)

# compute Sparse regression: sequential least squares
def sindy(lam,D,dxdt): 
    Xi = np.matmul(np.linalg.pinv(D), dxdt.T) # initial guess: Least-squares
    for k in range(10):
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
# Adding noise:
eps = 0.01      # noise strength
dxdt = np.column_stack(( xdt + np.random.normal(0, eps*np.std(xdt), len(xdt)) ))

# %
lam = 0.3      # lam is our sparsification knob.
Xi_drift = sindy(lam,D1,dxdt)

theta1 = np.zeros(10)
theta1[[1,8]] = 1,1
error = np.sum((theta1 - Xi_drift)**2)/np.sum(theta1**2)/100
print(error)

print(Xi_drift)

# %%
"""
For the diffusion identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# Adding noise:
eps = 0.01
xdts = np.column_stack(( xbt + np.random.normal(0, eps*np.std(xbt), len(xbt)) ))

lam = 0.45
Xi_diff = sindy(lam,D1,xdts)

theta2 = np.zeros(10)
theta2[[2,9]] = 0.5,0.5
error = np.sum((theta2 - Xi_diff)**2)/np.sum(theta2**2)/100
print(error)

print(Xi_diff)

# %%
def sparse_sol(xinit, params):
    T, xi_drift, xi_diff, polyorder, harmonic = params
    
    dt = 0.001
    t = np.arange(0, T+dt, dt)
    Nsamp = 500
    
    # np.random.seed(2021)
    y1 = []
    for ensemble in range(Nsamp):
        x0 = np.array([xinit])
        x = x0
        for n in range(len(t)-1):
            dW1 = np.sqrt(dt)*np.random.randn(1)
            
            D, nl = utils_library.library(np.vstack(x0), polyorder, harmonic)
            a = np.dot(D, xi_drift).reshape(-1)
            b = np.dot(D, xi_diff).reshape(-1)
            b = np.sqrt(b)
            
            sol = x0 + a*dt + b*dW1
            x0 = sol
            x = np.append(x, x0)
        y1.append(x)       
    y1 = np.array(y1)
    return np.mean(y1, axis=0)

# %%
""" The predictions """

T, x0 = 4, 1 # initial displacement for Bouc-Wen
dt = 0.001
t = np.arange(0, T+dt, dt)
np.random.seed(0)
_, _, y, _ = utils_response.blackscholes(x0, T)

# integrate true and identified systems
np.random.seed(0)
params = [T, Xi_drift, Xi_diff, 6, 0]
xt_pred = sparse_sol(x0, params)
