"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). 
-- 'A sparse Bayesian framework for discovering interpretable nonlinear stochastic
    dynamical systems with Gaussian white noise'.
    (in Arxiv: 'Learning governing physics from output only measurements')
   
*** This code is for figure generation.

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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from spyder_kernels.utils.iofuncs import load_dictionary
import numpy as np

"""
Black-Scholes equation ::
-----------------------------------------------------------------
"""

def blackscholes(x1, T, dt, params):
    # parameters of Black-Scholes equation:
    # ---------------------------------------------
    lam1, lam2, mu1, mu2 = params
    
    xzero = x1
    dt1 = dt
    t1 = np.arange(0, T+dt1, dt1)
    Nsamp = 200
    
    delmat = np.row_stack(([np.sqrt(dt1), 0],[(dt1**1.5)/2, (dt1**1.5)/(2*np.sqrt(3))]))
    
    y = []
    xz = []
    xzs = []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        # print(ensemble)
        xold = xzero
        xnew = xzero
        
        for k in range(1, len(t1)):
            delgen = np.dot(delmat, np.random.randn(2))
            dB = delgen[0]
            
            sol = xold+ (lam1*xold + lam2*np.abs(xold))*dt1 + np.sqrt(mu1*xold**2 + mu2*xold*np.abs(xold))*dB
            xold = sol
            xnew = np.append(xnew, sol)
            
        y.append(xnew)
    
        zint = xnew[0:-1]
        xfinal = xnew[1:] 
        xmz = (xfinal - zint) # 'x(t)-z' vector
        xmz2 = np.multiply(xmz, xmz)
        xz.append(xmz)
        xzs.append(xmz2)
        
    xz = pow(dt1,-1)*np.mean(np.array(xz), axis = 0)
    xzs = pow(dt1,-1)*np.mean(np.array(xzs), axis = 0)
    
    y = np.mean(np.array(y), axis=0)
    time = t1[0:-1]
    
    return xz, xzs, y, time    


# %%
np.random.seed(0)

""" The Actual system """
# System parameters:
lam1, lam2 = 1, 1
mu1, mu2 = 0.5, 0.5
sys = [lam1, lam2, mu1, mu2]

T, dt = 4, 0.001
x1 = 1 # initial displacement for Bouc-Wen
t = np.arange(0,T+dt,dt)
_, _, xt, _ = blackscholes(x1, T, dt, sys)


# %%
np.random.seed(0)

""" The identified system """
# System parameters:
lam1, lam2 = 1.09707, 0.95179
mu1, mu2 = 0.46917, 0.53116
sys = [lam1, lam2, mu1, mu2]

T, dt = 4, 0.001
x1 = 1 # initial displacement for Bouc-Wen
t = np.arange(0,T+dt,dt)
_, _, xt_i, _ = blackscholes(x1, T, dt, sys)


# %%
np.random.seed(0)

""" The Lower bound system """
y = []
for i in range(100):
    print(i)
    # System parameters:
    lam1, lam2 = np.random.normal(1.09707,0.06515,1), np.random.normal(0.95179,0.06515,1)
    mu1, mu2 = np.random.normal(0.46917,0.1599,1), np.random.normal(0.53116,0.1599)
    sys = [lam1, lam2, mu1, mu2]
    
    T, dt = 4, 0.001
    x1 = 1 # initial displacement for Bouc-Wen
    t = np.arange(0,T+dt,dt)
    _, _, xt_l, _ = blackscholes(x1, T, dt, sys)
    y.append(xt_l)
y = np.array(y)

# %%
xt_ii = np.mean(y, axis=0)
xt_l = xt_i - 2*np.std(y, axis=0)
xt_u = xt_i + 2*np.std(y, axis=0)

# %%
data_dict = load_dictionary('sindy_data/s_sindy_blackscholes.spydata')
xt_pred = data_dict[0]['xt_pred']

# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 26

fig1 = plt.figure(figsize=(10,6))
ax = fig1.add_subplot()
plt.plot(t, xt, color='r', label='True' )
plt.plot(t, xt_i, ':', color='k', linewidth=2, label='Proposed')
plt.plot(t, xt_pred, 'b-.', linewidth = 1.5, label='SINDy')
plt.fill_between(t, xt_l, xt_u, alpha = .5, color = 'c')
ax.add_patch(Rectangle((0, 0), 1, 10000, color="grey", alpha = 0.15))
ax.add_patch(Rectangle((1, 0), 3, 10000, color="y", alpha = 0.1))
plt.text(0.1, 200, "Training", fontsize=22)
plt.text(1.1, 200, "Prediction", fontsize=22)
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel(' $X$(t) ', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.legend(loc=1, columnspacing=0.7, handlelength=0.9, handletextpad=0.25, fontsize=22)
plt.margins(0)
# plt.xlim([0,6])

tp1, tp2 = 500, 1501
# location for the zoomed portion 
sub_axes1 = plt.axes([.2, .45, .4, .4]) 
# plot the zoomed portion
sub_axes1.plot(t[tp1:tp2], xt[tp1:tp2], color='r')
sub_axes1.plot(t[tp1:tp2], xt_i[tp1:tp2], '--', c = 'k')
sub_axes1.plot(t[tp1:tp2], xt_pred[tp1:tp2], 'b-.', linewidth = 1.5, label='SINDy')
sub_axes1.fill_between(t[tp1:tp2], xt_l[tp1:tp2], xt_u[tp1:tp2], alpha = .5, color = 'c')
sub_axes1.margins(0)
