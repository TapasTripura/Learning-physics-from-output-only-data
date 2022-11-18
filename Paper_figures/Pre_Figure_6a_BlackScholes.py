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
data_dict = load_dictionary('response_data/Response_Bscholes_a.spydata')
y = data_dict[0]['y']
t = data_dict[0]['t']
xt = data_dict[0]['xt']
xt_i = data_dict[0]['xt_i']

# %%
xt_ii = np.mean(y, axis=0)
xt_l = np.mean(y, axis=0) - 2*np.std(y, axis=0)
xt_u = np.mean(y, axis=0) + 2*np.std(y, axis=0)

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

tp1, tp2 = 500, 1501
# location for the zoomed portion 
sub_axes1 = plt.axes([.2, .45, .4, .4]) 
# plot the zoomed portion
sub_axes1.plot(t[tp1:tp2], xt[tp1:tp2], color='r')
sub_axes1.plot(t[tp1:tp2], xt_i[tp1:tp2], '--', c = 'k')
sub_axes1.plot(t[tp1:tp2], xt_pred[tp1:tp2], 'b-.', linewidth = 1.5, label='SINDy')
sub_axes1.plot(t[tp1:tp2], xt_i[tp1:tp2], ':', c = 'k', linewidth=2)
sub_axes1.fill_between(t[tp1:tp2], xt_l[tp1:tp2], xt_u[tp1:tp2], alpha = .5, color = 'c')
sub_axes1.margins(0)
