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
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from spyder_kernels.utils.iofuncs import load_dictionary

# %%
data_dict = load_dictionary('actual_data/Saved_data_BoucWen.spydata')
Zmeandrift = data_dict[0]['Zmeandrift']
Zmeandiff = data_dict[0]['Zmeandiff']
thetadrift = data_dict[0]['thetadrift']
thetadiff = data_dict[0]['thetadiff']
nl = data_dict[0]['nl']


# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

figure1=plt.figure(figsize = (18, 8))
plt.subplot(211)
xr = np.array(range(nl))
plt.stem(xr, Zmeandrift, use_line_collection = True, linefmt='blue', markerfmt ='bo', basefmt="k")
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=4)
plt.ylabel('PIP (drift)', fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.grid(True); plt.ylim(0,1.05)
plt.xticks(fontweight='bold');
plt.yticks(fontweight='bold');
plt.text(0, 1.1, "1", color='b', fontsize=20)
plt.text(0.8, 1.1, "X$_1$", color='b', fontsize=20)
plt.text(1.8, 1.1, "X$_2$", color='b', fontsize=20)
plt.text(2.8, 1.1, "X$_1^2$", color='b', fontsize=20)
plt.text(4.8, 1.1, "X$_2^2$", color='b', fontsize=20)
plt.text(5.8, 1.1, "X$_1^3$", color='b', fontsize=20)
plt.text(10, 1.1, "X$_1^4$", color='b', fontsize=20)
plt.text(27, 1.1, "sgn(X$_1$)", color='b', fontsize=20)
plt.text(29.5, 1.1, "|X$_1$|", color='b', fontsize=20)
plt.text(30.8, 1.1, "|X$_2$|", color='b', fontsize=20)
plt.text(32.2, 1.1, "X$_1$|X$_2$|", color='b', fontsize=20)

plt.text(1, -0.15, "1", fontsize=20, fontweight='bold')
plt.text(2, -0.15, "2", fontsize=20, fontweight='bold')
plt.text(3, -0.15, "3", fontsize=20, fontweight='bold')
plt.text(6, -0.15, "6", fontsize=20, fontweight='bold')
plt.text(28, -0.15, "28", fontsize=20, fontweight='bold')
plt.text(31, -0.15, "31", fontsize=20, fontweight='bold')
plt.text(32, -0.15, "32", fontsize=20, fontweight='bold')

plt.subplot(212)
xr = np.array(range(nl))
plt.stem(xr, Zmeandiff, use_line_collection = True, linefmt='blue', markerfmt ='bD', basefmt="k")
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=4)
plt.xlabel('Library functions', fontweight='bold');
plt.ylabel('PIP (diffusion)', fontweight='bold');
plt.grid(True); plt.ylim(0,1.05)
plt.xticks(fontweight='bold');
plt.yticks(fontweight='bold');
plt.title('(b)', fontweight='bold')
plt.text(0.2, 0.9, "1", color='b', fontsize=20)
plt.text(1, 0.6, "X$_1$", color='b', fontsize=20)
plt.text(30, 0.6, "|X|", color='b', fontsize=20)
plt.text(1, -0.15, "1", fontsize=24, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
""" All in all plots of theta """
figure2=plt.figure(figsize = (18, 12))
figure2.subplots_adjust(wspace=0.001)  # adjust space between axes
plt.subplot(331)
xy = np.vstack([thetadrift[0,:], thetadrift[1,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[0,:], thetadrift[1,:], c=z, s=100, cmap='icefire_r')
plt.xlabel(' 'r'$\theta (1)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.colorbar()
plt.xticks(fontweight='bold');plt.yticks(fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.subplot(332)
xy = np.vstack([thetadrift[2,:], thetadrift[3,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[2,:], thetadrift[3,:], c=z, s=100, cmap='icefire_r')
plt.xlabel(' 'r'$\theta (x_2)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1^2)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.colorbar()
plt.xlim(-20.2,-19.8)
plt.xticks(fontweight='bold');plt.yticks(fontweight='bold');
plt.title('(b)', fontweight='bold')
plt.subplot(333)
xy = np.vstack([thetadrift[5,:], thetadrift[6,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[5,:], thetadrift[6,:], c=z, s=100, cmap='icefire_r')
plt.xlabel(' 'r'$\theta (x_2^2)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1^3)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.colorbar()
plt.xticks(fontweight='bold');plt.yticks(fontweight='bold');
plt.title('(c)', fontweight='bold')
plt.subplot(334)
xy = np.vstack([thetadrift[10,:], thetadrift[28,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[10,:], thetadrift[28,:], c=z, s=100, cmap='icefire_r')
plt.xlabel(' 'r'$\theta (x_1^4)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (sgn(x_1)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.colorbar()
plt.xticks(fontweight='bold');plt.yticks(fontweight='bold');
plt.title('(d)', fontweight='bold')
plt.subplot(335)
xy = np.vstack([thetadrift[30,:], thetadrift[31,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[30,:], thetadrift[31,:], c=z, s=100, cmap='icefire_r')
plt.xlabel(' 'r'$\theta (|x_1|)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (|x_2|)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.colorbar()
plt.xticks(fontweight='bold');plt.yticks(fontweight='bold');
plt.title('(e)', fontweight='bold')
plt.subplot(336)
xy = np.vstack([thetadrift[30,:], thetadrift[32,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[30,:], thetadrift[32,:], c=z, s=100, cmap='icefire_r')
plt.xlabel(' 'r'$\theta (|x_1|)_{drift}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1|x_1|)_{drift}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.colorbar()
plt.xticks(fontweight='bold');plt.yticks(fontweight='bold');
plt.title('(f)', fontweight='bold')

plt.subplot(337)
xy = np.vstack([thetadiff[0,:], thetadiff[1,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadiff[0,:], thetadiff[1,:], c=z, s=100, cmap='icefire_r')
plt.xlabel(' 'r'$\theta (1)_{diffusion}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (x_1)_{diffusion}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.xticks(fontweight='bold');plt.yticks(fontweight='bold');
plt.title('(I)', fontweight='bold')
plt.colorbar()
plt.subplot(338)
xy = np.vstack([thetadiff[0,:], thetadiff[30,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadiff[0,:], thetadiff[30,:], c=z, s=100, cmap='icefire_r')
plt.xlabel(' 'r'$\theta (1)_{diffusion}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (|x_1|)_{diffusion}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.xticks(fontweight='bold');plt.yticks(fontweight='bold');
plt.title('(II)', fontweight='bold')
plt.colorbar()
plt.subplot(339)
xy = np.vstack([thetadiff[1,:], thetadiff[30,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadiff[1,:], thetadiff[30,:], c=z, s=100, cmap='icefire_r')
plt.xlabel(' 'r'$\theta (x_1)_{diffusion}$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (|x_1|)_{diffusion}$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.xticks(fontweight='bold');plt.yticks(fontweight='bold');
plt.title('(III)', fontweight='bold')
plt.colorbar() 
plt.tight_layout()
plt.show()
