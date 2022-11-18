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
from spyder_kernels.utils.iofuncs import load_dictionary
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

# %%
data_dict = load_dictionary('response_data/Response_BoucWen.spydata')
y1 = data_dict[0]['y1']
y2 = data_dict[0]['y2']
t = data_dict[0]['t']
dt = data_dict[0]['dt']
xt = data_dict[0]['xt']
xt_i = data_dict[0]['xt_i']

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
