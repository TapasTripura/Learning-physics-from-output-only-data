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

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

# %%
"""
A Bouc-Wen oscillator excited by random noise
--> Partially oberved variable Z(t)
--------------------------------------------------------------------------
"""

def boucwen(x1, x2, x3, T, dt, sys):
    # Actual parameters of Bouc-Wen oscillator in Equation
    # ----------------------------------------------------
    m, c, k, lam, A1, A2, A3, nbar, sigma1 = sys
    t = np.arange(0, T+dt, dt)
    Nsamp = 100 #int(1/dt) # no. of samples in the run
    
    np.random.seed(2021)
    force = np.random.randn(len(t), Nsamp)
    
    y1, y2, y3 = [], [], []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        # print(ensemble)
        x0 = np.array([x1, x2, x3])
        x = np.vstack(x0)  # Zero initial condition.
        for n in range(len(t)-1):
            dW = force[n,ensemble]
            
            a1 = x0[1]
            a2 = -(c/m)*x0[1]-(k/m)*lam*x0[0]-(k/m)*(1-lam)*x0[2]
            a3 = -A1*x0[2]*np.abs(x0[1])*pow(np.abs(x0[2]),nbar-1) \
                -A2*x0[1]*pow(np.abs(x0[2]),nbar) + A3*x0[2]
            b1 = 0
            b2 = (2*sigma1/m)
            b3 = 0
    
            sol1 = x0[0] + a1*dt 
            sol2 = x0[1] + a2*dt + b2*dW*np.sqrt(dt)
            sol3 = x0[2] + a3*dt 
            x0 = np.array([sol1, sol2, sol3])
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
    
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    
    y = np.array([np.mean(y1, axis=0), np.mean(y2, axis=0)])
    time = t[0:-1]   
    return y, time   # hysteresis z(t) is unkown,


def boucwen_i(x1, x2, T, dt, nsamp, sys):
    # Identified parameters of Bouc-Wen oscillator in Equation
    # --------------------------------------------------------
    ai, bi, ci, di, ei, fi, gi, hi, ii, ji, ki, li, mi, ni, oi, pi, sigma = sys
    
    t = np.arange(0, T+dt, dt)
    Nsamp = nsamp #int(1/dt) # no. of samples in the run
    
    np.random.seed(2021)
    force = np.random.randn(len(t), Nsamp)
    
    xnew1 = []
    xnew2 = []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        xn0 = np.array([x1, x2])
        xn = np.vstack(xn0)
    
        for n in range(len(t)-1):     
            dW = force[n,ensemble]
            soln1 = xn0[0] + xn0[1]*dt 
            soln2 = xn0[1] + (ai +bi*xn0[0] +ci*xn0[1] +di*(xn0[0]**2) \
                              +ei*(xn0[1]**2) +fi*(xn0[0]**3) +gi*(xn0[0]**4) \
                                  # +hi*(xn0[0]**5) +ii*(xn0[0]**6) \
                                      +ji*np.sign(xn0[0]) +ki*np.abs(xn0[0]) \
                                          +li*np.abs(xn0[1]) \
                                              +mi*xn0[0]*np.abs(xn0[0]) )*dt \
                + sigma*np.sqrt(ni +oi*xn0[0] +pi*np.abs(xn0[0]))*dW*np.sqrt(dt)
            xn0 = np.array([soln1, soln2])
            xn = np.column_stack((xn, xn0))

        xnew1.append(xn[0,:])
        xnew2.append(xn[1,:])
    
    xnew1 = np.mean(np.array(xnew1), axis = 0)
    xnew1 = xnew1 + np.abs(np.mean(xnew1, axis = 0))
    xnew2 = np.mean(np.array(xnew2), axis = 0)
    
    y = np.row_stack((xnew1, xnew2))
    time = t[0:-1]    
    return y, time


# %%
""" The actual system """

# Actual parameters
# ---------------------------------------------
m, c, k, lam = 1, 20, 10000, 0.5
A1, A2, A3, nbar = 0.5, 0.5, 1, 3
sigma1 = 3

sys1 = [m, c, k, lam, A1, A2, A3, nbar, sigma1]

# Response generation:
T, dt = 500, 0.002
t = np.arange(0, T+dt, dt)
x1, x2, x3 = 0, 0, 0 # initial displacement for Bouc-Wen
xt, _ = boucwen(x1, x2, x3, T, dt, sys1)


# %%
""" The identified system """

# Identified parameters
# ---------------------------------------------
ai, bi, ci = -46.4, -5246, -20
di, ei, fi = -614, -0.05, -2328
gi, hi, ii = 1829, 361.2, 154
ji, ki, li = 10.64, 159, 1.625
mi, ni, oi = 1439, 4.65, -4.872
pi, sigma1 = 6.786, 3
    
sys2 = [ai, bi, ci, di, ei, fi, gi, hi, ii, ji, ki, li, mi, ni, oi, pi, sigma1]

# Response generation:
T, dt = 500, 0.002
t = np.arange(0, T+dt, dt)
Nsamp = 100
x1, x2 = 0, 0 # initial displacement for Bouc-Wen
xt_i, _ = boucwen_i(x1, x2, T, dt, Nsamp, sys2)


# %%
""" The bounded system """

y1, y2 = [], []
for i in range(500):
    print(i)
    # # Identified parameters
    # # ---------------------------------------------
    ai, bi, ci = normal(-46.4,0.1495), normal(-5246,26.9518), normal(-20,0.0462)
    di, ei, fi = normal(-614,104.033), normal(-0.05,0.0178), normal(-2328,501.634)
    gi, hi, ii = normal(1829,487.381), normal(361.2,842.879), normal(154,971.077)
    ji, ki, li = normal(10.64,0.7264), normal(159,20.4214), normal(1.625,0.268)
    mi, ni, oi = normal(1439,214.361), normal(4.65,0.1495), normal(-4.872,6.6353)
    pi = normal(6.786,8.5098)
    
    sys2 = [ai, bi, ci, di, ei, fi, gi, hi, ii, ji, ki, li, mi, ni, oi, pi, sigma1]
    
    # # Response generation:
    T, dt = 500, 0.002
    t = np.arange(0, T+dt, dt)
    Nsamp = 100
    x1, x2 = 0, 0 # initial displacement for Bouc-Wen
    xt_l, _ = boucwen_i(x1, x2, T, dt, Nsamp, sys2)
    y1.append(xt_l[0,:])
    y2.append(xt_l[1,:])
y1 = np.array(y1)
y2 = np.array(y2)

# %%
xt_ii = np.array([np.nanmean(y1, axis = 0), np.nanmean(y2, axis=0)])
xt_l = np.array([np.nanmean(y1, axis = 0)-2*np.std(y1, axis=0), np.nanmean(y2, axis = 0)-2*np.std(y2, axis=0)])
xt_u = np.array([np.nanmean(y1, axis = 0)+2*np.std(y1, axis=0), np.nanmean(y2, axis = 0)+2*np.std(y2, axis=0)])

# %%
fig1 = plt.figure(figsize=(14,8))
plt.plot(t, xt[0,:], color='r', label='True' )
plt.plot(t, xt_ii[0,:], ':', color='k', label='Mean')
plt.fill_between(t[:], xt_u[0,:], xt_l[0,:], alpha = .5, color = 'b')
plt.xlabel('Time (s)')
plt.ylabel(' $X$(t) ')
plt.legend(loc=1)
plt.margins(0)

tp1, tp2 = 1400, 1501
# location for the zoomed portion 
sub_axes1 = plt.axes([.22, .67, .2, .2]) 
# plot the zoomed portion
sub_axes1.plot(t[tp1:tp2], xt[0,tp1:tp2], color='r')
sub_axes1.plot(t[tp1:tp2], xt_ii[0,tp1:tp2], ':', c = 'k', linewidth=2)
sub_axes1.fill_between(t[tp1:tp2], xt_l[0,tp1:tp2], xt_u[0,tp1:tp2], alpha = .5, color = 'b')
sub_axes1.margins(0)

# %%
fig2 = plt.figure(figsize=(14,8))
plt.plot(t, xt[1,:], color='r', label='True' )
plt.plot(t, xt_ii[1,:], ':', color='k', label='Identified')
plt.fill_between(t, xt_u[1,:], xt_l[1,:], alpha = .5, color = 'b')
plt.xlabel('Time (s)')
plt.ylabel(' $\dot{X}$(t) ')
plt.legend(loc=1)
plt.margins(0)

tp1, tp2 = 1400,1501
# location for the zoomed portion 
sub_axes1 = plt.axes([.22, .2, .2, .2]) 
# plot the zoomed portion
sub_axes1.plot(t[tp1:tp2], xt[1,tp1:tp2], c = 'r')
sub_axes1.plot(t[tp1:tp2], xt_ii[1,tp1:tp2], '--', c = 'k', linewidth=2)
sub_axes1.fill_between(t[tp1:tp2], xt_l[1,tp1:tp2], xt_u[1,tp1:tp2], alpha = .5, color = 'b')
sub_axes1.margins(0)

# %%
"""
Broken Axis plot ...
"""
figure2, ax = plt.subplots(2,4, figsize=(20, 10), sharey='row', facecolor='w')
figure2.subplots_adjust(wspace=0.1, hspace=0.4)  # adjust space between axes
x1range = [51, 201, 301, 491]; x2range = [51.5, 201.5, 301.5, 491.5];
figure2.text(0.5, 0.01, "Time (s)", fontweight='bold', fontsize=30)

# plot the same data on both axes
for i in range(len(ax)*len(ax[0])):
    if i<4:
        ax[0,i].plot(t[int(x1range[i]/dt):int(x2range[i]/dt)], xt[0,int(x1range[i]/dt):int(x2range[i]/dt)], 'r', linewidth=2)
        ax[0,i].plot(t[int(x1range[i]/dt):int(x2range[i]/dt)], xt_ii[0,int(x1range[i]/dt):int(x2range[i]/dt)], ':k', linewidth=4)
        ax[0,i].set_xlim(x1range[i], x2range[i])
        ax[0,i].set_ylim(-.008, 0.008)
        ax[0,i].tick_params(axis='x', labelrotation=45)
        ax[0,i].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
        ax[0,i].fill_between(t[int(x1range[i]/dt):int(x2range[i]/dt)], \
                             xt_u[0,int(x1range[i]/dt):int(x2range[i]/dt)], \
                                 xt_l[0,int(x1range[i]/dt):int(x2range[i]/dt)], alpha = 0.8, color = 'orange')
        ax[0,i].grid(True)
    else:
        ax[1,i-4].plot(t[int(x1range[i-4]/dt):int(x2range[i-4]/dt)], xt[1,int(x1range[i-4]/dt):int(x2range[i-4]/dt)], 'r', linewidth=2)
        ax[1,i-4].plot(t[int(x1range[i-4]/dt):int(x2range[i-4]/dt)], xt_ii[1,int(x1range[i-4]/dt):int(x2range[i-4]/dt)], ':k', linewidth=4)
        ax[1,i-4].set_xlim(x1range[i-4], x2range[i-4])
        ax[1,i-4].set_ylim(-.5, .5)
        ax[1,i-4].tick_params(axis='x', labelrotation=45)
        ax[1,i-4].ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
        ax[1,i-4].fill_between(t[int(x1range[i-4]/dt):int(x2range[i-4]/dt)], \
                               xt_u[1,int(x1range[i-4]/dt):int(x2range[i-4]/dt)], \
                                   xt_l[1,int(x1range[i-4]/dt):int(x2range[i-4]/dt)], alpha = 0.8, color = 'orange')
        ax[1,i-4].grid(True)
        
# hide the spines between ax and ax2
ax[0,0].spines['right'].set_visible(False)
ax[1,0].spines['right'].set_visible(False)
ax[0,-1].spines['left'].set_visible(False)
ax[1,-1].spines['left'].set_visible(False)
ax[0,0].yaxis.tick_left(); ax[1,0].yaxis.tick_left()
ax[0,-1].yaxis.tick_right(); ax[1,-1].yaxis.tick_right()
ax[0,-1].yaxis.set_ticks_position('right'); ax[1,-1].yaxis.set_ticks_position('right')
ax[0,0].set_ylabel(' 'r'${\mathbb{E}[\mathbf{X}]}$', fontweight='bold')
ax[1,0].set_ylabel(' 'r'${\mathbb{E}[\mathbf{\dot{X}}]}$', fontweight='bold');
ax[0,-1].legend(['Actual system','Identified system'], ncol=2, loc=1)

for i in range(1, len(ax[0])-1):
    ax[0,i].spines['right'].set_visible(False)
    ax[0,i].spines['left'].set_visible(False)
    ax[1,i].spines['right'].set_visible(False)
    ax[1,i].spines['left'].set_visible(False)
    
for i in range(0, len(ax[0])-1):
    d = .02 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax[0,i].transAxes, color='k', clip_on=False)
    ax[0,i].plot((1-d,1+d), (-d,+d), **kwargs)
    ax[0,i].plot((1-d,1+d),(1-d,1+d), **kwargs)
    kwargs.update(transform=ax[0,i+1].transAxes)  # switch to the bottom axes
    ax[0,i+1].plot((-d,+d), (1-d,1+d), **kwargs)
    ax[0,i+1].plot((-d,+d), (-d,+d), **kwargs)
    
    kwargs = dict(transform=ax[1,i].transAxes, color='k', clip_on=False)
    ax[1,i].plot((1-d,1+d), (-d,+d), **kwargs)
    ax[1,i].plot((1-d,1+d),(1-d,1+d), **kwargs)
    kwargs.update(transform=ax[1,i+1].transAxes)  # switch to the bottom axes
    ax[1,i+1].plot((-d,+d), (1-d,1+d), **kwargs)
    ax[1,i+1].plot((-d,+d), (-d,+d), **kwargs)
plt.show()
