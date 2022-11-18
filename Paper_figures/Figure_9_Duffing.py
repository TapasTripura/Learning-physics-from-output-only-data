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

# %%

"""
A Duffing Van der pol system excited by random noise
----------------------------------------------------------------------
"""
def duffing(x1, x2, T, dt, params):
    # parameters of Duffing oscillator in Equation
    m, c, k, k3, sigma = params
    
    # solution by Taylor 1.5 strong scheme Run with dt=0.01
    # -------------------------------------------------------
    t = np.arange(0, T+dt, dt)
    Nsamp = 500 # no. of samples in the run
    delmat = np.row_stack(([np.sqrt(dt), 0],[(dt**1.5)/2, (dt**1.5)/(2*np.sqrt(3))]))
    
    y1 = []
    y2 = []
    xz = []
    xzs = []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        x0 = np.array([x1, x2])
        x = np.vstack(x0)  # Zero initial condition.
        for n in range(len(t)-1):
            delgen = np.dot(delmat, np.random.normal(0,1,2))
            dW = delgen[0]
            a1 = x0[1]
            a2 = -(c/m)*x0[1]-(k/m)*x0[0]-(k3/m)*(x0[0]**3)
            b2 = (sigma/m)*x0[0]

            sol1 = x0[0] + a1*dt 
            sol2 = x0[1] + a2*dt + b2*dW 
            x0 = np.array([sol1, sol2])
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        
        zint = x[1,0:-1]
        xfinal = x[1,1:] 
        xmz = (xfinal - zint) # 'x(t)-z' vector
        xmz2 = np.multiply(xmz, xmz)
        xz.append(xmz)
        xzs.append(xmz2)
        
    xz = pow(dt,-1)*np.mean(np.array(xz), axis = 0)
    xzs = pow(dt,-1)*np.mean(np.array(xzs), axis = 0)
    
    y1 = np.array(y1)
    y2 = np.array(y2)
    y = np.array([np.mean(y1, axis=0), np.mean(y2, axis=0)])
    time = t[0:-1]
    
    return xz, xzs, y, time


# %%
np.random.seed(0)

""" The actual system """
# System parameters:
m = 1
c = 2
k = 1000
k3 = 100000
sigma = 10 # 1 (paper)

sys = [m, c, k, k3, sigma]

# Response generation:
T = 5
dt= 0.001
t = np.arange(0, T+dt, dt)
x1, x2 = 0.05, 0 # initial displacement for duffing
_, _, xt, _ = duffing(x1, x2, T, dt, sys)


# %%
np.random.seed(0)

""" The identified system """
# System parameters:
m = 1
c = 1.9789
k = 1002.37
k3 = 98657.5
sigma = 10.31 # 1 (paper)

sys = [m, c, k, k3, sigma]

# Response generation:
T = 5
dt= 0.001
t = np.arange(0, T+dt, dt)
x1, x2 = 0.05, 0 # initial displacement for duffing
_, _, xt_i, _ = duffing(x1, x2, T, dt, sys)


# %%
np.random.seed(0)

""" The bounded system """
y1, y2 = [], []
for i in range(500):
    print(i)
    """ The lower bound system """
    # System parameters:
    m = 1
    c = np.random.normal(1.9789,0.0396)
    k = np.random.normal(1002.37,0.0205)
    k3 = np.random.normal(98657.5,1402.65)
    sigma = np.random.normal(10.31,2.3825) 
    
    sys = [m, c, k, k3, sigma]
    
    # Response generation:
    T = 5
    dt= 0.001
    t = np.arange(0, T+dt, dt)
    x1, x2 = 0.05, 0 # initial displacement for duffing
    _, _, xt_l, _ = duffing(x1, x2, T, dt, sys)
    y1.append(xt_l[0,:])
    y2.append(xt_l[1,:])
y1 = np.array(y1)
y2 = np.array(y2)

# %%
xt_ii = np.array([np.nanmean(y1, axis = 0), np.nanmean(y2, axis=0)])
xt_l = np.array([np.nanmean(y1, axis = 0)-2*np.std(y1, axis=0), np.nanmean(y2, axis = 0)-2*np.std(y2, axis=0)])
xt_u = np.array([np.nanmean(y1, axis = 0)+2*np.std(y1, axis=0), np.nanmean(y2, axis = 0)+2*np.std(y2, axis=0)])

# %%
data_dict = load_dictionary('sindy_data/s_sindy_dvp_10s.spydata')
xt_pred = data_dict[0]['xt_pred']

# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 26

fig1 = plt.figure(figsize=(10,6))
ax = fig1.add_subplot()
plt.plot(t, xt[0,:], color='r', label='True' )
plt.plot(t, xt_ii[0,:], ':', color='k', linewidth=1.5, label='Proposed')
plt.plot(t, xt_pred[0,:5001], 'b-.', label='SINDy')

plt.fill_between(t[:-50], xt_u[0,:-50], xt_l[0,:-50], alpha = .75, color = 'orange')
ax.add_patch(Rectangle((0, -0.06), 1, 0.12, color="grey", alpha = 0.1))
ax.add_patch(Rectangle((1, -0.06), 4, 0.12, color="c", alpha = 0.1))
plt.text(0.1, -0.055, "Training", fontsize=22)
plt.text(1.1, -0.055, "Prediction", fontsize=22)
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel(' $X$(t) ')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.legend(loc=1, ncol=1, columnspacing=0.7, handlelength=0.9, handletextpad=0.25, fontsize=24)
plt.margins(0)

tp1, tp2 = 2000, 2201
# location for the zoomed portion 
sub_axes1 = plt.axes([.62, .2, .25, .2]) 
# plot the zoomed portion
sub_axes1.plot(t[tp1:tp2], xt[0,tp1:tp2], color='r')
sub_axes1.plot(t[tp1:tp2], xt_ii[0,tp1:tp2], ':', c = 'k')
sub_axes1.plot(t[tp1:tp2], xt_pred[0,tp1:tp2], 'b-.')
sub_axes1.fill_between(t[tp1:tp2], xt_l[0,tp1:tp2], xt_u[0,tp1:tp2], alpha = .5, color = 'orange')
sub_axes1.margins(0)

# %%
fig2 = plt.figure(figsize=(10,6))

ax = fig2.add_subplot()
plt.plot(t, xt[1,:], color='r', label='True' )
plt.plot(t, xt_ii[1,:], ':', color='k', linewidth=1.5, label='Proposed')
plt.plot(t, xt_pred[1,:5001], 'b-.', label='SINDy')

plt.fill_between(t, xt_u[1,:], xt_l[1,:], alpha = .75, color = 'orange')
ax.add_patch(Rectangle((0, -2), 1, 4, color="grey", alpha = 0.1))
ax.add_patch(Rectangle((1, -2), 4, 4, color="c", alpha = 0.1))
plt.text(0.1, -1.75, "Training", fontsize=22)
plt.text(1.1, -1.75, "Prediction", fontsize=22)
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel(' $\dot{X}$(t) ')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.legend(loc=1, ncol=1, columnspacing=0.7, handlelength=0.9, handletextpad=0.25, fontsize=24)
plt.margins(0)

tp1, tp2 = 2000, 2201
# location for the zoomed portion 
sub_axes1 = plt.axes([.62, .2, .25, .2]) 
# plot the zoomed portion
sub_axes1.plot(t[tp1:tp2], xt[1,tp1:tp2], c = 'r')
sub_axes1.plot(t[tp1:tp2], xt_ii[1,tp1:tp2], ':', c = 'k')
sub_axes1.plot(t[tp1:tp2], xt_pred[1,tp1:tp2], 'b-.')
sub_axes1.fill_between(t[tp1:tp2], xt_l[1,tp1:tp2], xt_u[1,tp1:tp2], alpha = .5, color = 'orange')
sub_axes1.margins(0)
