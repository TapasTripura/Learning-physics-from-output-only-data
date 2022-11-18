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

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from spyder_kernels.utils.iofuncs import load_dictionary

# %%
data_dict = load_dictionary('actual_data/Saved_data_Duffing.spydata')
Zmeandrift = data_dict[0]['Zmeandrift']
Zmeandiff = data_dict[0]['Zmeandiff']
thetadrift = data_dict[0]['thetadrift']
thetadiff = data_dict[0]['thetadiff']
mutdiff = data_dict[0]['mutdiff']
nl = data_dict[0]['nl']

# %%
"""
Plotting commands:
"""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

figure1=plt.figure(figsize = (14, 10))
plt.subplot(211)
xr = np.array(range(nl))
plt.stem(xr, Zmeandrift, use_line_collection = True, linefmt='blue', basefmt="k")
plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.ylabel('PIP (drift)', fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.grid(True); plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(212)
xr = np.array(range(nl))
plt.stem(xr, Zmeandiff, use_line_collection = True, linefmt='blue', basefmt="k")
plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.xlabel('Library functions', fontweight='bold');
plt.ylabel('PIP (diffusion)', fontweight='bold');
plt.grid(True); plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.title('(b)', fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Post processing:
less_uncertain = 0.02
mutind = np.where(mutdiff<(np.max(np.abs(mutdiff))*less_uncertain))
Zmeandiff[mutind] = Zmeandiff[mutind]*0
thetanor = thetadiff[3,:] - np.abs(thetadrift[1,:])
munormal = np.mean(thetanor)

figure2=plt.figure(figsize = (16, 12))
plt.subplot(221)
xy = np.vstack([thetadrift[1,:], thetadrift[2,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[1,:], thetadrift[2,:], c=z, s=100, cmap='turbo')
plt.colorbar()
plt.xlabel(' 'r'$\theta (x_1)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_2)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(a)', fontweight='bold')
plt.xlim(-1010,-995)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(222)
xy = np.vstack([thetadrift[1,:], thetadrift[6,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[1,:], thetadrift[6,:], c=z, s=100, cmap='turbo')
plt.colorbar(format='%.e')
plt.xlabel(' 'r'$\theta (x_1)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1^3)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(b)', fontweight='bold')
plt.xlim(-1010,-995); plt.ylim(-103000,-94000);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(223)
xy = np.vstack([thetadrift[2,:], thetadrift[6,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[2,:], thetadrift[6,:], c=z, s=100, cmap='turbo')
plt.colorbar(format='%.e')
plt.xlabel(' 'r'$\theta (x_2)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1^3)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(c)', fontweight='bold')
plt.ylim(-103000,-94000)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(224)
ax=sns.distplot(thetanor, kde_kws={"color": "b"},  hist_kws={"color": "r"})
plt.xlabel(' 'r'$\theta (x_1^2)_{diffusion}$', fontweight='bold'); 
plt.title('(d)', fontweight='bold'); plt.xlim(90,120);
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.tight_layout()
plt.show()
