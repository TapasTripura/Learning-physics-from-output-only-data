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
from spyder_kernels.utils.iofuncs import load_dictionary
import matplotlib.gridspec as gridspec
import seaborn as sns

# %%
data_dict1 = load_dictionary('actual_data/Saved_data_2DOF_drift.spydata')
data_dict2 = load_dictionary('actual_data/Saved_data_2DOF_diffusion.spydata')
nl = data_dict1[0]['nl']
Zmeandrift1 = data_dict1[0]['Zmeandrift1']
Zmeandrift2 = data_dict1[0]['Zmeandrift2']
thetadrift1 = data_dict1[0]['thetadrift1']
thetadrift2 = data_dict1[0]['thetadrift2']

thetadiff11 = data_dict2[0]['thetadiff11']
thetadiff22 = data_dict2[0]['thetadiff22']
Zmeandiff11 = data_dict2[0]['Zmeandiff11']
Zmeandiff22 = data_dict2[0]['Zmeandiff22']

# %%
"""
Plotting Command
"""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

figure1=plt.figure(figsize = (14, 10))
plt.subplot(211)
xr = np.array(range(nl))
plt.stem(xr, Zmeandrift1, use_line_collection = True, linefmt='blue', basefmt="k")
plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.ylabel('PIP (drift)', fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.grid(True); plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(212)
xr = np.array(range(nl))
plt.stem(xr, Zmeandrift2, use_line_collection = True, linefmt='blue', basefmt="k")
plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.xlabel('Library functions', fontweight='bold');
plt.ylabel('PIP (diffusion)', fontweight='bold');
plt.grid(True); plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.title('(b)', fontweight='bold')
plt.tight_layout()
plt.show()

# %%
figure2=plt.figure(figsize = (18, 14))
plt.subplot(331)
xy = np.vstack([thetadrift1[1,:], thetadrift1[2,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[1,:], thetadrift1[2,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_2)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(a)', fontweight='bold')
plt.xlim(-6010, -5990)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(332)
xy = np.vstack([thetadrift1[1,:], thetadrift1[3,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[1,:], thetadrift1[3,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(b)', fontweight='bold')
plt.xlim(-6010, -5990); plt.ylim(1980, 2020)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(333)
xy = np.vstack([thetadrift1[1,:], thetadrift1[4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[1,:], thetadrift1[4,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(c)', fontweight='bold')
plt.xlim(-6010, -5990)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(334)
xy = np.vstack([thetadrift1[1,:], thetadrift1[36,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[1,:], thetadrift1[36,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (sgn(y_2))$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(d)', fontweight='bold')
plt.xlim(-6010, -5990)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(335)
xy = np.vstack([thetadrift1[2,:], thetadrift1[3,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[2,:], thetadrift1[3,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(e)', fontweight='bold')
plt.ylim(1980, 2020)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(336)
xy = np.vstack([thetadrift1[2,:], thetadrift1[4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[2,:], thetadrift1[4,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(f)', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(337)
xy = np.vstack([thetadrift1[2,:], thetadrift1[36,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[2,:], thetadrift1[36,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (sgn(y_2))$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(g)', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(338)
xy = np.vstack([thetadrift1[3,:], thetadrift1[4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[3,:], thetadrift1[4,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_3)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(h)', fontweight='bold')
plt.xlim(1980, 2020)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(339)
xy = np.vstack([thetadrift1[3,:], thetadrift1[36,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift1[3,:], thetadrift1[36,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_3)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (sgn(y_2))$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(i)', fontweight='bold')
plt.xlim(1980,2020)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.tight_layout()
plt.show()

# %%
figure3=plt.figure(figsize = (20, 10))
plt.subplot(231)
xy = np.vstack([thetadrift2[1,:], thetadrift2[2,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift2[1,:], thetadrift2[2,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_2)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(a)', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(232)
xy = np.vstack([thetadrift2[1,:], thetadrift2[3,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift2[1,:], thetadrift2[3,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(b)', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(233)
xy = np.vstack([thetadrift2[1,:], thetadrift2[4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift2[1,:], thetadrift2[4,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(c)', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(234)
xy = np.vstack([thetadrift2[2,:], thetadrift2[3,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift2[2,:], thetadrift2[3,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(d)', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(235)
xy = np.vstack([thetadrift2[2,:], thetadrift2[4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift2[2,:], thetadrift2[4,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(e)', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.subplot(236)
xy = np.vstack([thetadrift2[3,:], thetadrift2[4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift2[3,:], thetadrift2[4,:], c=z, s=100, cmap='rainbow')
plt.colorbar()
plt.xlabel(' 'r'$\theta (y_3)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(f)', fontweight='bold')
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.tight_layout()
plt.show()

# %%
figure4=plt.figure(figsize = (18 , 6))
plt.subplot(121)
sns.distplot(thetadiff11[0,:], kde_kws={"color": "b"},  hist_kws={"color": "g"})
plt.xlabel(' 'r'$\theta (1)_{drift}$', fontweight='bold'); 
plt.title('(a)', fontweight='bold')
plt.xlim([95,107])
plt.subplot(122)
sns.distplot(thetadiff22[0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
plt.xlabel(' 'r'$\theta (1)_{diffusion}$', fontweight='bold'); 
plt.title('(b)', fontweight='bold')
plt.xlim([98,106])
plt.tight_layout()
plt.show()
