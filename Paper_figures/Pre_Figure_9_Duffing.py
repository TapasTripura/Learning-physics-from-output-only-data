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
data_dict = load_dictionary('response_data/Response_Duffing.spydata')
y1 = data_dict[0]['y1']
y2 = data_dict[0]['y2']
t = data_dict[0]['t']
xt = data_dict[0]['xt']
xt_i = data_dict[0]['xt_i']

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
