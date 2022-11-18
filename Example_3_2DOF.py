"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). 
-- 'A sparse Bayesian framework for discovering interpretable nonlinear stochastic
    dynamical systems with Gaussian white noise'.
    (in Arxiv: 'Learning governing physics from output only measurements')
   
*** This code is for the Example 3: 2DOF system.

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
from numpy import linalg as LA
from numpy.random import gamma as IG
from numpy.random import beta
from numpy.random import binomial as bern
import utils_library
import utils_gibbs
import utils_response
from numpy.random import multivariate_normal as mvrv
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
import seaborn as sns

# %%
"""
MCMC sampling for model updating
"""
def sparse_stoc(xdts, ydata, polyorder, modfun, harmonic, MCMC, burn_in):
    # Ensemble mean Dictionary Creation:
    if len(ydata) == 1: # for 1-D SDEs,
        libr = []
        for j in range(len(ydata[0])):
            data = np.column_stack((ydata[0][j,0:-1]))
            Dtemp, nl = utils_library.library(data, polyorder, modfun, harmonic)
            libr.append(Dtemp)
        libr = np.array(libr)
        D = np.mean(libr, axis = 0)
    elif len(ydata) == 2: # for 2-D SDEs/ 1-DOF systems,
        libr = []
        for j in range(len(ydata[0])):
            data = np.row_stack((ydata[0][j,0:-1], ydata[1][j,0:-1]))
            Dtemp, nl = utils_library.library(data, polyorder, modfun, harmonic)
            libr.append(Dtemp)
        libr = np.array(libr)
        D = np.mean(libr, axis = 0)
    elif len(ydata) == 3: # for 3-D SDEs/ like Lorenz systems,
        libr = []
        for j in range(len(ydata[0])):
            data = np.row_stack((ydata[0][j,0:-1], ydata[1][j,0:-1], ydata[2][j,0:-1]))
            Dtemp, nl = utils_library.library(data, polyorder, modfun, harmonic)
            libr.append(Dtemp)
        libr = np.array(libr)
        D = np.mean(libr, axis = 0)
    elif len(ydata) == 4: # for 4-D SDEs/ 2-DOF systems,
        libr = []
        for j in range(len(ydata[0])):
            data = np.row_stack((ydata[0][j,0:-1], ydata[1][j,0:-1], ydata[2][j,0:-1], ydata[3][j,0:-1]))
            Dtemp, nl = utils_library.library(data, polyorder, modfun, harmonic)
            libr.append(Dtemp)
        libr = np.array(libr)
        D = np.mean(libr, axis = 0)
        
    # Adding 1% of the std. of acceleration as noise:
    # xdts = xdts + 0.01*np.random.normal(0, 0.01*np.std(xdts), len(xdts))
    
    # Residual variance:
    err_var = utils_library.res_var(D, xdts)
    
    """
    # Gibbs sampling:
    """
    # Hyper-parameters
    ap, bp = 0.1, 1 # for beta prior for p0
    av, bv = 0.5, 0.5 # inverge gamma for vs
    asig, bsig = 1e-4, 1e-4 # invese gamma for sig^2
    
    # Parameter Initialisation:
    p0 = np.zeros(MCMC)
    vs = np.zeros(MCMC)
    sig = np.zeros(MCMC)
    p0[0] = 0.1
    vs[0] = 10
    sig[0] = err_var
    
    N = len(xdts)
    
    # Initial latent vector
    zval = np.zeros(nl)
    zint  = utils_library.latent(nl, D, xdts)
    zstore = np.transpose(np.vstack([zint]))
    zval = zint
    
    zval0 = zval
    vs0 = vs[0]
    mu, BSIG, Aor, index = utils_gibbs.sigmu(zval0, D, vs0, xdts)
    Sz = sum(zval)
    
    # Sample theta from Normal distribution
    thetar = mvrv(mu, np.dot(sig[0], BSIG))
    thetat = np.zeros(nl)
    thetat[index] = thetar
    theta = np.vstack(thetat)
    
    print('MCMC has started: ')
    for i in range(1, MCMC):
        if i % 50 == 0:
            print('MCMC: ', i)
        
        # sample z from the Bernoulli distribution:
        zr = np.zeros(nl) # instantaneous latent vector (z_i):
        zr = zval
        for j in range(nl):
            ztemp0 = zr
            ztemp0[j] = 0
            if np.mean(ztemp0) == 0:
                PZ0 = utils_gibbs.pyzv0(xdts, N, asig, bsig)
            else:
                vst0 = vs[i-1]
                PZ0 = utils_gibbs.pyzv(D, ztemp0, vst0, N, xdts, asig, bsig)
            
            ztemp1 = zr
            ztemp1[j] = 1      
            vst1 = vs[i-1]
            PZ1 = utils_gibbs.pyzv(D, ztemp1, vst1, N, xdts, asig, bsig)
            
            zeta = PZ0 - PZ1  
            zeta = p0[i-1]/( p0[i-1] + np.exp(zeta)*(1-p0[i-1]))
            zr[j] = bern(1, p = zeta, size = None)
        
        zval = zr
        zstore = np.append(zstore, np.vstack(zval), axis = 1)
        
        # sample sig^2 from inverse Gamma:
        asiggamma = asig+0.5*N
        temp = np.matmul(np.matmul(mu.T, LA.inv(BSIG)), mu)
        bsiggamma = bsig+0.5*(np.dot(xdts.T, xdts) - temp)
        sig[i] = 1/IG(asiggamma, 1/bsiggamma) # inverse gamma RVs
        
        # sample vs from inverse Gamma:
        avvs = av+0.5*Sz
        bvvs = bv+(np.matmul(np.matmul(thetar.T, LA.inv(Aor)), thetar))/(2*sig[i])
        vs[i] = 1/IG(avvs, 1/bvvs) # inverse gamma RVs
        
        # sample p0 from Beta distribution:
        app0 = ap+Sz
        bpp0 = bp+nl-Sz # Here, P=nl (no. of functions in library)
        p0[i] = beta(app0, bpp0)
        # or, np.random.beta()
        
        # Sample theta from Normal distribution:
        vstheta = vs[i]
        mu, BSIG, Aor, index = utils_gibbs.sigmu(zval, D, vstheta, xdts)
        Sz = sum(zval)
        thetar = mvrv(mu, np.dot(sig[i], BSIG))
        thetat = np.zeros(nl)
        thetat[index] = thetar
        theta = np.append(theta, np.vstack(thetat), axis = 1)
    
    zstore = zstore[:, burn_in:]    
    Zmean = np.mean(zstore, axis=1)
    
    theta= theta[:, burn_in:]
    mut = np.mean(theta, axis=1)
    sigt = np.cov(theta, bias = False)
    
    return zstore, Zmean, theta, mut, sigt, D

# %%
"""
--->>> Identification of Drifts
"""
# Response generation:
T = 1
x1, x2, x3, x4 = 0.5, 0, 0.05, 0 # initial displacement for 2dof
xdrift1, xdrift2, _, _, _, y1, y2, y3, y4, t_eval = utils_response.dof2sys(x1, x2, x3, x4, T)
xdt = [xdrift1, xdrift2]
ydata = [y1, y2, y3, y4]

polyorder, modulus, harmonic, MCMC, burn_in = [6, 1, 0, 4000, 1000]

zstoredrift, Zmeandrift, thetadrift, mutdrift, sigtdrift = [], [], [], [], []
for i in range(len(xdt)):
    print('state-{}'.format(i))
    zstore, Zmean, theta, mut, sigt, D = \
        sparse_stoc(xdt[i], ydata, polyorder, modulus, harmonic, MCMC, burn_in)
    zstoredrift.append(zstore)
    Zmeandrift.append(Zmean)
    thetadrift.append(theta)
    mutdrift.append(mut)
    sigtdrift.append(sigt)

# Post processing:
for i in range(len(xdt)): 
    Zmeandrift[i][np.where(Zmeandrift[i] < 0.5)] = 0
    mutdrift[i][np.where(Zmeandrift[i] < 0.5)] = 0

# %%
"""
--->>> Diffusion, GAMMA_{ij}, identification
"""
# Response generation:
x1, x2, x3, x4 = 0.001, 0, 0.001, 0 # initial displacement for 2dof
_, _, xdiff11, _, xdiff22, y1, y2, y3, y4, t_eval = utils_response.dof2sys(x1, x2, x3, x4, T)
bxt = [xdiff11, xdiff22]
ydata = [y1, y2, y3, y4]

zstorediff, Zmeandiff, thetadiff, mutdiff, sigtdiff = [], [], [], [], []
for i in range(len(bxt)):
    print('State-{}'.format(i))
    zstore, Zmean, theta, mut, sigt, D = \
        sparse_stoc(bxt[i], ydata, polyorder, modulus, harmonic, MCMC, burn_in)
    zstorediff.append(zstore)
    Zmeandiff.append(Zmean)
    thetadiff.append(theta)
    mutdiff.append(mut)
    sigtdiff.append(sigt)
    
# Post processing:
for i in range(len(xdt)): 
    Zmeandiff[i][np.where(Zmeandiff[i] < 0.6)] = 0
    mutdiff[i][np.where(Zmeandiff[i] < 0.6)] = 0
    Zmeandiff[i][np.where(np.diag(sigtdiff[i])>100)] = 0
    mutdiff[i][np.where(np.diag(sigtdiff[i])>100)] = 0

# %%
"""
Plotting Command
"""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24

figure1=plt.figure(figsize = (14, 10))
plt.subplot(211)
plt.stem(np.arange(0,len(Zmeandrift[0])), Zmeandrift[0], use_line_collection = True, linefmt='blue', basefmt="k")
plt.axhline(y= 0.5, color='r', linestyle='-.')
plt.ylabel('PIP (drift)', fontweight='bold');
plt.title('(a)', fontweight='bold')
plt.grid(True); plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(212)
plt.stem(np.arange(0,len(Zmeandrift[1])), Zmeandrift[1], use_line_collection = True, linefmt='blue', basefmt="k")
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
xy = np.vstack([thetadrift[0][1,:], thetadrift[0][2,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[0][1,:], thetadrift[0][2,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_2)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(a)', fontweight='bold')
plt.grid(True); plt.xlim(-6010, -5990)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(332)
xy = np.vstack([thetadrift[0][1,:], thetadrift[0][3,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[0][1,:], thetadrift[0][3,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(b)', fontweight='bold')
plt.grid(True); plt.xlim(-6010, -5990); plt.ylim(1980, 2020)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(333)
xy = np.vstack([thetadrift[0][1,:], thetadrift[0][4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[0][1,:], thetadrift[0][4,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(c)', fontweight='bold')
plt.grid(True); plt.xlim(-6010, -5990)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(334)
xy = np.vstack([thetadrift[0][1,:], thetadrift[0][211,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[0][1,:], thetadrift[0][211,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (sgn(y_2))$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(d)', fontweight='bold')
plt.grid(True); plt.xlim(-6010, -5990)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(335)
xy = np.vstack([thetadrift[0][2,:], thetadrift[0][3,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[0][2,:], thetadrift[0][3,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(e)', fontweight='bold')
plt.grid(True); plt.ylim(1980, 2020)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(336)
xy = np.vstack([thetadrift[0][2,:], thetadrift[0][4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[0][2,:], thetadrift[0][4,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(f)', fontweight='bold')
plt.grid(True); 
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(337)
xy = np.vstack([thetadrift[0][2,:], thetadrift[0][211,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[0][2,:], thetadrift[0][211,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (sgn(y_2))$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(g)', fontweight='bold')
plt.grid(True); plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(338)
xy = np.vstack([thetadrift[0][3,:], thetadrift[0][4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[0][3,:], thetadrift[0][4,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_3)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(h)', fontweight='bold')
plt.grid(True); plt.xlim(1980, 2020); plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(339)
xy = np.vstack([thetadrift[0][3,:], thetadrift[0][211,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[0][3,:], thetadrift[0][211,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_3)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (sgn(y_2))$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(i)', fontweight='bold')
plt.grid(True); plt.xlim(1980,2020); plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.tight_layout()
plt.show()

# %%
figure3=plt.figure(figsize = (18, 12))
plt.subplot(231)
xy = np.vstack([thetadrift[1][1,:], thetadrift[1][2,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[1][1,:], thetadrift[1][2,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_2)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(a)', fontweight='bold')
plt.grid(True); plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(232)
xy = np.vstack([thetadrift[1][1,:], thetadrift[1][3,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[1][1,:], thetadrift[1][3,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(b)', fontweight='bold')
plt.grid(True); plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(233)
xy = np.vstack([thetadrift[1][1,:], thetadrift[1][4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[1][1,:], thetadrift[1][4,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_1)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(c)', fontweight='bold')
plt.grid(True); plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(234)
xy = np.vstack([thetadrift[1][2,:], thetadrift[1][3,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[1][2,:], thetadrift[1][3,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_3)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(d)', fontweight='bold')
plt.grid(True); plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(235)
xy = np.vstack([thetadrift[1][2,:], thetadrift[1][4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[1][2,:], thetadrift[1][4,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_2)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(e)', fontweight='bold')
plt.grid(True); plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

plt.subplot(236)
xy = np.vstack([thetadrift[1][3,:], thetadrift[1][4,:]])
z = gaussian_kde(xy)(xy)
plt.scatter(thetadrift[1][3,:], thetadrift[1][4,:], c=z, s=100)
plt.xlabel('Drift- 'r'$\theta (y_3)$', fontweight='bold'); 
plt.ylabel(' 'r'$\theta (y_4)$', fontweight='bold');
plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
plt.title('(f)', fontweight='bold')
plt.grid(True); plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.tight_layout()
plt.show()

# %%
figure4=plt.figure(figsize = (14, 8))
gs = gridspec.GridSpec(2, 6)

ax1 = plt.subplot(gs[0, :4])
ax1.stem(np.arange(0,len(Zmeandiff[0]),1), Zmeandiff[0], use_line_collection = True, linefmt='blue', basefmt="k")
ax1.axhline(y= 0.5, color='r', linestyle='-.')
ax1.set_xlabel('Library functions (Drift)', fontweight='bold');
ax1.set_ylabel('PIP (drift)', fontweight='bold');
ax1.set_title('(a)', fontweight='bold')
ax1.grid(True); plt.ylim(0,1.05)
 
ax2 = plt.subplot(gs[1, :4])
ax2.stem(np.arange(0,len(Zmeandiff[0]),1), Zmeandiff[1], use_line_collection = True, linefmt='blue', basefmt="k")
ax2.axhline(y= 0.5, color='r', linestyle='-.')
ax2.set_xlabel('Library functions (Diffusion)', fontweight='bold');
ax2.set_ylabel('PIP (diffusion)', fontweight='bold');
ax2.grid(True); plt.ylim(0,1.05)
ax2.set_title('(c)', fontweight='bold')
 
ax3 = plt.subplot(gs[0, 4:6])
ax=sns.distplot(thetadiff[0][0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
ax3.set_xlabel('Diffusion- 'r'$\theta (1)$', fontweight='bold'); 
ax3.set_title('(b)', fontweight='bold')
ax3.grid(True); 

ax4 = plt.subplot(gs[1, 4:6])
ax=sns.distplot(thetadiff[1][0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
ax4.set_xlabel('Diffusion- 'r'$\theta (1)$', fontweight='bold'); 
ax4.set_title('(d)', fontweight='bold')
ax4.grid(True); 

plt.tight_layout()
plt.show()

# %%
figure5=plt.figure(figsize = (18 , 6))
plt.subplot(121)
sns.distplot(thetadiff[0][0,:], kde_kws={"color": "b"},  hist_kws={"color": "g"})
plt.xlabel(' 'r'$\theta (1)_{drift}$', fontweight='bold'); 
plt.title('(a)', fontweight='bold')
plt.grid(True); 
plt.subplot(122)
sns.distplot(thetadiff[1][0,:], kde_kws={"color": "b"},  hist_kws={"color": "r"})
plt.xlabel(' 'r'$\theta (1)_{diffusion}$', fontweight='bold'); 
plt.title('(b)', fontweight='bold')
plt.grid(True); 

plt.tight_layout()
plt.show()
