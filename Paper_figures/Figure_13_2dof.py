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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from spyder_kernels.utils.iofuncs import load_dictionary

# %%

"""
2-DOF system with top-linear and bottom base isolated ::
-----------------------------------------------------------------
"""

def dof2sys(x1, x2, x3, x4, T, dt, sys):
    # parameters of 2-DOF in Equation
    # ---------------------------------------------
    m1, m2, k1, k2, c1, c2, mu, gf, sigma1, sigma2 = sys
    
    # solution by Taylor 1.5 strong scheme Run with dt=0.01
    # -------------------------------------------------------
    t = np.arange(0, T+dt, dt)
    Nsamp = 200 #int(1/dt) # no. of samples in the run
    
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    xz1 = []
    xz2 = []
    xzs11 = []
    xzs12 = []
    xzs22 = []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        x0 = np.array([x1, x2, x3, x4])
        x = x0  # Zero initial condition.
        for n in range(len(t)-1):
            dW = np.sqrt(dt)*np.random.randn(2)
            dW1 = dW[0]
            dW2 = dW[1]
            
            a1 = x0[1]
            a2 = -(c1/m1)*x0[1]-mu*gf*np.sign(x0[1])-(k1/m1)*x0[0] \
                -(c2/m1)*(x0[1]-x0[3])-(k2/m1)*(x0[0]-x0[2])
            a3 = x0[3]
            a4 = -(c2/m2)*(x0[3]-x0[1])-(k2/m2)*(x0[2]-x0[0])
            b1 = 0
            b2 = (sigma1)/m1
            b3 = 0
            b4 = (sigma2)/m2
            
            sol1 = x0[0] + a1*dt
            sol2 = x0[1] + a2*dt + b2*dW1
            sol3 = x0[2] + a3*dt
            sol4 = x0[3] + a4*dt + b4*dW2
            
            x0 = np.array([sol1, sol2, sol3, sol4])
            x = np.column_stack((x, x0))
            
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
        y4.append(x[3,:])
        
        zint1 = x[1, 0:-1]
        xfinal1 = x[1, 1:] 
        xmz1 = (xfinal1 - zint1) # 'x1(t)-z1' vector
        zint2 = x[3, 0:-1]
        xfinal2 = x[3, 1:] 
        xmz2 = (xfinal2 - zint2) # 'x2(t)-z2' vector
        
        xz1.append(xmz1)
        xz2.append(xmz2)
        
        xmzsq11 = np.multiply(xmz1, xmz1)
        xzs11.append(xmzsq11)
        
        xmzsq12 = np.multiply(xmz1, xmz2)
        xzs12.append(xmzsq12)
        
        xmzsq22 = np.multiply(xmz2, xmz2)
        xzs22.append(xmzsq22)
        
    xz1 = pow(dt,-1)*np.mean(np.array(xz1), axis = 0)
    xz2 = pow(dt,-1)*np.mean(np.array(xz2), axis = 0)
    
    xzs11 = pow(dt,-1)*np.mean(np.array(xzs11), axis = 0)
    xzs12 = pow(dt,-1)*np.mean(np.array(xzs12), axis = 0)
    xzs22 = pow(dt,-1)*np.mean(np.array(xzs22), axis = 0)
    
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    y = np.array([np.mean(y1, axis=0), np.mean(y2, axis=0), np.mean(y3, axis=0), np.mean(y4, axis=0)])
    time = t[0:-1]
    
    return xz1, xz2, xzs11, xzs12, xzs22, y, time


# %%
np.random.seed(0)

""" The actual system """

# System parameters:
m1, m2 = 1, 1
k1, k2 = 4000, 2000
c1, c2 = 2, 2
mu, gf = 1, 9.81
sigma1, sigma2 = 10, 10

sys = [m1, m2, k1, k2, c1, c2, mu, gf, sigma1, sigma2]

# Response generation:
T = 5
dt= 0.001
t = np.arange(0, T+dt, dt)

# Response generation:
x1, x2, x3, x4 = 0.5, 0, 0.05, 0 # initial displacement for 2dof
_, _, _, _, _, xt, _ = dof2sys(x1, x2, x3, x4, T, dt, sys)


# %%
np.random.seed(0)

""" The identified system """

# System parameters:
m1, m2 = 1, 1
k1, k2 = 3999.65, 2000.65
c1, c2 = 1.975, 1.9935
mu, gf = 1, 9.8935
sigma1, sigma2 = np.sqrt(100.681), np.sqrt(102.198)

sys = [m1, m2, k1, k2, c1, c2, mu, gf, sigma1, sigma2]

# Response generation:
T = 5
dt= 0.001
t = np.arange(0, T+dt, dt)

# Response generation:
x1, x2, x3, x4 = 0.5, 0, 0.05, 0 # initial displacement for 2dof
_, _, _, _, _, xt_i, _ = dof2sys(x1, x2, x3, x4, T, dt, sys)


# %%
np.random.seed(0)

""" The bounded system """

y1, y2, y3, y4 = [], [], [], []
for i in range(500):
    print(i)
    # System parameters:
    m1, m2 = 1, 1
    k1, k2 = np.random.normal(3999.65,15.2311), np.random.normal(2000.65,15.2311)
    c1, c2 = np.random.normal(1.975,0.0119), np.random.normal(1.9935,0.0119)
    mu, gf = 1, np.random.normal(9.8935)
    sigma1, sigma2 = np.sqrt(np.random.normal(100.681,1.7965)), np.sqrt(np.random.normal(102.198,1.1182))
    
    sys = [m1, m2, k1, k2, c1, c2, mu, gf, sigma1, sigma2]
    
    # Response generation:
    T = 5
    dt= 0.001
    t = np.arange(0, T+dt, dt)
    
    # Response generation:
    x1, x2, x3, x4 = 0.5, 0, 0.05, 0 # initial displacement for 2dof
    _, _, _, _, _, xt_l, _ = dof2sys(x1, x2, x3, x4, T, dt, sys)
    y1.append(xt_l[0,:])
    y2.append(xt_l[1,:])
    y3.append(xt_l[2,:])
    y4.append(xt_l[3,:])
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y4 = np.array(y4)

# %%
xt_ii = np.array([np.mean(y1, axis=0), np.mean(y2, axis=0), np.mean(y3, axis=0), np.mean(y4, axis=0)])

xt_l = np.array([np.mean(y1, axis = 0)-2*np.std(y1, axis=0), np.mean(y2, axis = 0)-2*np.std(y2, axis=0), \
                 np.mean(y3, axis = 0)-2*np.std(y3, axis=0), np.mean(y4, axis = 0)-2*np.std(y4, axis=0)])
    
xt_u = np.array([np.mean(y1, axis = 0)+2*np.std(y1, axis=0), np.mean(y2, axis = 0)+2*np.std(y2, axis=0), \
                np.mean(y3, axis = 0)+2*np.std(y3, axis=0), np.mean(y4, axis = 0)+2*np.std(y4, axis=0)])

# %%
data_dict = load_dictionary('sindy_data/s_sindy_2DOF.spydata')
xt_pred = data_dict[0]['xt_pred']

# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 26

fig1 = plt.figure(figsize=(10,6))
ax = fig1.add_subplot()
plt.plot(t[:4002], xt[0,:4002], color='r', label='True' )
plt.plot(t[:4002], xt_i[0,:4002], ':', color='k', label='Proposed')
plt.plot(t[:4002], xt_pred[0,:], 'b', linewidth=2, label='SINDy')
plt.fill_between(t[:4002], xt_u[0,:4002], xt_l[0,:4002], alpha = .5, color = 'orange')
ax.add_patch(Rectangle((0, -20), 1, 40, color="grey", alpha = 0.1))
ax.add_patch(Rectangle((1, -20), 3, 40, color="r", alpha = 0.1))
plt.text(0.1, -10, "Training", fontsize=22)
plt.text(1.1, -10, "Prediction", fontsize=22)
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel(' $X_1$(t) ', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.legend(loc=3, ncol=3, columnspacing=0.7, handlelength=0.9, handletextpad=0.25, fontsize=22)
plt.margins(0)

tp1, tp2 = 1000, 1101
# location for the zoomed portion 
sub_axes1 = plt.axes([.2, .65, .25, .2]) 
# plot the zoomed portion
sub_axes1.plot(t[tp1:tp2], xt[0,tp1:tp2], c = 'r')
sub_axes1.plot(t[tp1:tp2], xt_i[0,tp1:tp2], ':', c = 'k')
plt.plot(t[tp1:tp2], xt_pred[0,tp1:tp2], 'b')
sub_axes1.fill_between(t[tp1:tp2], xt_l[0,tp1:tp2], xt_u[0,tp1:tp2], alpha = .75, color = 'orange')
sub_axes1.margins(0)

# %%
fig2 = plt.figure(figsize=(10,6))
ax2 = fig2.add_subplot()
plt.plot(t[:4002], xt[1,:4002], color='r', label='True' )
plt.plot(t[:4002], xt_i[1,:4002], ':', color='k', label='Proposed')
plt.plot(t[:4002], xt_pred[1,:4002], 'b', linewidth=2, label='SINDy')
plt.fill_between(t[:4002], xt_u[1,:4002], xt_l[1,:4002], alpha = .5, color = 'orange')
ax2.add_patch(Rectangle((0, -1500), 1, 3000, color="grey", alpha = 0.15))
ax2.add_patch(Rectangle((1, -1500), 3, 3000, color="r", alpha = 0.1))
plt.text(0.1, -800, "Training", fontsize=20)
plt.text(1.1, -800, "Prediction", fontsize=20)
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel(' $\dot{X}_1$(t) ', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.legend(loc=3, ncol=3, columnspacing=0.7, handlelength=0.9, handletextpad=0.25, fontsize=22)
plt.margins(0)

tp1, tp2 = 1000, 1101
# location for the zoomed portion 
sub_axes1 = plt.axes([.2, .65, .25, .2]) 
# plot the zoomed portion
sub_axes1.plot(t[tp1:tp2], xt[1,tp1:tp2], c = 'r')
sub_axes1.plot(t[tp1:tp2], xt_i[1,tp1:tp2], ':', c = 'k')
sub_axes1.plot(t[tp1:tp2], xt_pred[1,tp1:tp2], 'b')
sub_axes1.fill_between(t[tp1:tp2], xt_l[1,tp1:tp2], xt_u[1,tp1:tp2], alpha = .75, color = 'orange')
sub_axes1.margins(0)

# %%
fig3 = plt.figure(figsize=(10,6))
ax3 = fig3.add_subplot()
plt.plot(t[:4002], xt[2,:4002], color='r', label='True' )
plt.plot(t[:4002], xt_ii[2,:4002], ':', color='k', label='Proposed')
plt.plot(t[:4002], xt_pred[2,:4002], 'b', linewidth=2, label='SINDy')
plt.fill_between(t[:4002], xt_u[2,:4002], xt_l[2,:4002], alpha = .5, color = 'orange')
ax3.add_patch(Rectangle((0, -8), 1, 16, color="grey", alpha = 0.15))
ax3.add_patch(Rectangle((1, -8), 3, 16, color="r", alpha = 0.1))
plt.text(0.1, -5, "Training", fontsize=20)
plt.text(1.1, -5, "Prediction", fontsize=20)
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel(' $X_2$(t) ', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.legend(loc=3, ncol=3, columnspacing=0.7, handlelength=0.9, handletextpad=0.25, fontsize=22)
plt.margins(0)

tp1, tp2 = 1000, 1101
# location for the zoomed portion 
sub_axes1 = plt.axes([.2, .65, .25, .2]) 
# plot the zoomed portion
sub_axes1.plot(t[tp1:tp2], xt[2,tp1:tp2], c = 'r')
sub_axes1.plot(t[tp1:tp2], xt_ii[2,tp1:tp2], ':', c = 'k')
sub_axes1.plot(t[tp1:tp2], xt_pred[2,tp1:tp2], 'b')
sub_axes1.fill_between(t[tp1:tp2], xt_l[2,tp1:tp2], xt_u[2,tp1:tp2], alpha = .5, color = 'orange')
sub_axes1.margins(0)

# %%
fig4 = plt.figure(figsize=(10,6))
ax4 = fig4.add_subplot()
plt.plot(t[:4002], xt[3,:4002], color='r', label='True' )
plt.plot(t[:4002], xt_i[3,:4002], ':', color='k', label='Proposed')
plt.plot(t[:4002], xt_pred[3,:4002], 'b', linewidth=2, label='SINDy')
plt.fill_between(t[:4002], xt_u[3,:4002], xt_l[3,:4002], alpha = .75, color = 'orange')
ax4.add_patch(Rectangle((0, -800), 1, 1600, color="grey", alpha = 0.15))
ax4.add_patch(Rectangle((1, -800), 3, 1600, color="r", alpha = 0.1))
plt.text(0.1, -500, "Training", fontsize=20)
plt.text(1.1, -500, "Prediction", fontsize=20)
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel(' $\dot{X}_2$(t) ', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.legend(loc=3, ncol=3, columnspacing=0.7, handlelength=0.9, handletextpad=0.25, fontsize=22)
plt.margins(0)

tp1, tp2 = 1000, 1101
# location for the zoomed portion 
sub_axes1 = plt.axes([.2, .65, .25, .2]) 
# plot the zoomed portion
sub_axes1.plot(t[tp1:tp2], xt[3,tp1:tp2], c = 'r')
sub_axes1.plot(t[tp1:tp2], xt_i[3,tp1:tp2], ':', c = 'k')
sub_axes1.plot(t[tp1:tp2], xt_pred[3,tp1:tp2], 'b')
sub_axes1.fill_between(t[tp1:tp2], xt_l[3,tp1:tp2], xt_u[3,tp1:tp2], alpha = .5, color = 'orange')
sub_axes1.margins(0)
