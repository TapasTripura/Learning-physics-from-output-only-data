"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). 
-- 'A sparse Bayesian framework for discovering interpretable nonlinear stochastic
    dynamical systems with Gaussian white noise'.
    (in Arxiv: 'Learning governing physics from output only measurements')
   
*** This code is a part of gibbs sampling.

# If you use any part of this code, then please cite us.

@article{tripura2022learning,
  title={Learning governing physics from output only measurements},
  author={Tripura, Tapas and Chakraborty, Souvik},
  journal={arXiv preprint arXiv:2208.05609},
  year={2022}
}

# The citation can also be found in Elsevier, MSSP journal.
"""

import numpy as np
from scipy.special import loggamma as LG
from numpy import linalg as LA

"""
Theta: Multivariate Normal distribution
"""
def sigmu(z, D, vs, xdts):
    index = np.array(np.where(z != 0))
    index = np.reshape(index,-1) # converting to 1-D array, 
    Dr = D[:,index] 
    Aor = np.eye(len(index)) # independent prior
    # Aor = np.dot(len(Dr), LA.inv(np.matmul(Dr.T, Dr))) # g-prior
    BSIG = LA.inv(np.matmul(Dr.T,Dr) + np.dot(pow(vs,-1), LA.inv(Aor)))
    mu = np.matmul(np.matmul(BSIG,Dr.T),xdts)
    return mu, BSIG, Aor, index

"""
P(Y|zi=(0|1),z-i,vs)
"""
def pyzv(D, ztemp, vs, N, xdts, asig, bsig):
    rind = np.array(np.where(ztemp != 0))[0]
    rind = np.reshape(rind, -1) # converting to 1-D array,   
    Sz = sum(ztemp)
    Dr = D[:, rind] 
    Aor = np.eye(len(rind)) # independent prior
    # Aor = np.dot(N, LA.inv(np.matmul(Dr.T, Dr))) # g-prior
    BSIG = np.matmul(Dr.T, Dr) + np.dot(pow(vs, -1),LA.inv(Aor))
    
    (sign, logdet0) = LA.slogdet(LA.inv(Aor))
    (sign, logdet1) = LA.slogdet(LA.inv(BSIG))
    
    PZ = LG(asig + 0.5*N) -0.5*N*np.log(2*np.pi) - 0.5*Sz*np.log(vs) \
        + asig*np.log(bsig) - LG(asig) + 0.5*logdet0 + 0.5*logdet1
    denom1 = np.eye(N) - np.matmul(np.matmul(Dr, LA.inv(BSIG)), Dr.T)
    denom = (0.5*np.matmul(np.matmul(xdts.T, denom1), xdts))
    PZ = PZ - (asig+0.5*N)*(np.log(bsig + denom))
    return PZ

"""
P(Y|zi=0,z-i,vs)
"""
def pyzv0(xdts, N, asig, bsig):
    PZ0 = LG(asig + 0.5*N) - 0.5*N*np.log(2*np.pi) + asig*np.log(bsig) - LG(asig) \
        + np.log(1) - (asig+0.5*N)*np.log(bsig + 0.5*np.matmul(xdts.T, xdts))
    return PZ0