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
data_dict = load_dictionary('response_data/Response_2DOF.spydata')
y1 = data_dict[0]['y1']
y2 = data_dict[0]['y2']
y3 = data_dict[0]['y3']
y4 = data_dict[0]['y4']

t = data_dict[0]['t']
xt = data_dict[0]['xt']
xt_i = data_dict[0]['xt_i']

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
