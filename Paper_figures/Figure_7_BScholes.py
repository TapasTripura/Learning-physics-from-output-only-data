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
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
from spyder_kernels.utils.iofuncs import load_dictionary

# %%
data_dict = load_dictionary('actual_data/Saved_data_Bscholes_Taylor15.spydata')
Zmeandrift = data_dict[0]['Zmeandrift']
Zmeandiff = data_dict[0]['Zmeandiff']
thetadrift = data_dict[0]['thetadrift']
thetadiff = data_dict[0]['thetadiff']
mutdiff = data_dict[0]['mutdiff']
nl = data_dict[0]['nl']

# %%
figure3=plt.figure(figsize = (20, 6))
gs = gridspec.GridSpec(1,3)

ax1 = plt.subplot(gs[0])
xr = np.array(range(nl))
plt.stem(xr, Zmeandrift, use_line_collection = True, linefmt='blue', basefmt="k", label='Drift')
plt.stem(xr, Zmeandiff, use_line_collection = True, linefmt='m', basefmt="k", markerfmt='Dm', label = 'Diffusion')
plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.xlabel('Library functions', fontweight='bold');
plt.ylabel('PIP', fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.legend()
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

ax1 = plt.subplot(gs[1])
ax=sns.distplot(thetadrift[1,:], kde_kws={"color": "Blue"},  hist_kws={"color": "r"})
plt.xlabel('Drift- 'r'$\theta (x)$', fontweight='bold'); 
plt.xlim(1.8, 2.5)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.title('(b)', fontweight='bold')

ax1 = plt.subplot(gs[2])
ax=sns.distplot(thetadiff[2,:], bins=400, kde_kws={"color": "g"},  hist_kws={"color": "m"})
plt.xlabel('Diffusion- 'r'$\theta (x^2)$', fontweight='bold'); 
plt.xlim(0.9, 1.1)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.title('(c)', fontweight='bold')
plt.tight_layout()
plt.show()
