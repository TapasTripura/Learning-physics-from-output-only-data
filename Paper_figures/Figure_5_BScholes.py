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
import pickle
import seaborn as sns

# %%
[x1with,x2with,x1no,x2no] = pickle.load(open("actual_data/data_Blackschole.p","rb"))

a1 = x1no[1,:][np.where(x1no[1,:]>0)]
a2 = x1no[8,:][np.where(x1no[8,:]>0)]

b1 = x2no[2,:][np.where(x2no[2,:]>0)]
b2 = x2no[9,:][np.where(x2no[9,:]>0)]

plt.rcParams["font.family"] = "Serif"
plt.rcParams['font.size'] = 20

figure1=plt.figure(figsize = (16, 10))  
figure1.subplots_adjust(hspace=4)  # adjust space between axes
figure1.text(0.34,0.98,'Library with modulus functions: |x| & x|x|', fontweight='bold')
figure1.text(0.32,0.49,'Library without modulus functions: |x| & x|x|', fontweight='bold')

plt.subplot(221)
ax=sns.distplot(a1, bins=100, kde_kws={"color": "r"}, \
                hist_kws={"alpha":0.25, "color":"r"}, label=''r'$\theta (x)$')
ax=sns.distplot(a2, bins=100, kde_kws={"color": "k"}, \
                hist_kws={"alpha":0.25, "color":"k"}, label=''r'$\theta (|x|)$')
plt.xlabel(' 'r'$\theta _{drift}$', fontweight='bold'); 
plt.xlim(-2, 6)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.legend(loc=1)

plt.subplot(222)
ax=sns.distplot(b1, bins=100, kde_kws={"color": "r"}, \
                hist_kws={"alpha":0.25, "color":"r"}, label=''r'$\theta (x^2)$')
ax=sns.distplot(b2, bins=100, kde_kws={"color": "k"}, \
                hist_kws={"alpha":0.25, "color":"k"}, label=''r'$\theta (x|x|)$')
plt.xlabel(' 'r'$\theta _{diffusion}$', fontweight='bold'); 
plt.xlim(-4, 6)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.title('(b)', fontweight='bold')
plt.legend()

plt.subplot(223)
ax=sns.distplot(x1with[1,:], bins=100, kde_kws={"color": "b"},  hist_kws={"color": "g"}, label=''r'$\theta (x)$')
plt.xlabel(''r'$\theta _{drift}$', fontweight='bold'); 
plt.xlim(1.6, 2.4)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.title('\n(c)', fontweight='bold')
plt.legend()

plt.subplot(224)
ax=sns.distplot(x2with[2,:], bins=500, kde_kws={"color": "b"}, hist_kws={"color": "g"}, label=''r'$\theta (x^2)$')
plt.xlabel(''r'$\theta _{diffusion}$', fontweight='bold'); 
plt.xlim(0.9, 1.2)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.title('\n(d)', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
