"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). 
-- 'A sparse Bayesian framework for discovering interpretable nonlinear stochastic
    dynamical systems with Gaussian white noise'.
    (in Arxiv: 'Learning governing physics from output only measurements')
   
*** This code is for the Example 4: Bouc-Wen, with partially observed state variables.

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
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

# %%
"""
For the drift identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Response generation:
T = 1
x1, x2, x3 = 0.55, 0, 0.005 # initial displacement for Bouc-Wen
xdt, bxt, y1, y2, t_eval = utils_response.boucwen(x1, x2, x3, T)

# Expected Dictionary Creation:
libr = []
for j in range(len(y1)):
    data = np.row_stack((y1[j,0:-1], y2[j,0:-1]))
    Dtemp, nl = utils_library.library(data, polyn=6, modulus=1, harmonic=0)
    libr.append(Dtemp)
libr = np.array(libr)
D= np.mean(libr, axis = 0)

xdts = xdt

# Adding 1% of the std. of acceleration as noise:
# xdts = xdts + np.random.normal(0, 0.01*np.std(xdts), len(xdts))

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
MCMC = 4000 # No. of samples in Markov Chain,
p0 = np.zeros(MCMC)
vs = np.zeros(MCMC)
sig = np.zeros(MCMC)
p0[0] = 0.1
vs[0] = 10
sig[0] = err_var
burnin = 1001 # < MCMC

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

for i in range(1, MCMC):
    print(i)
    
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

"""
Post-processing of data:
"""
# Marginal posterior inclusion probabilities (PIP):
zstoredrift = zstore[:, burnin:]
Zmeandrift = np.mean(zstoredrift, axis=1)

# Post processing:
thetadrift = theta[:, burnin:]
mutdrift = np.mean(thetadrift, axis=1)
sigtdrift = np.cov(thetadrift, bias = False)


"""
For the diffusion identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Response generation:
T = 1
x1, x2, x3 = 0.001, 0, 0.005 # initial displacement for Bouc-Wen
xdt, xbt, y1, y2, t_eval = utils_response.boucwen(x1, x2, x3, T)

# Dictionary Creation:
libr = []
for j in range(len(y1)):
    data = np.row_stack((y1[j,0:-1], y2[j,0:-1]))
    Dtemp, nl = utils_library.library(data, polyn=6, modulus=1, harmonic=0)
    libr.append(Dtemp)
libr = np.array(libr)
D = np.mean(libr, axis = 0)

xdts = xbt

# Adding 1% of the std. of acceleration as noise:
# xdts = xdts + np.random.normal(0, 0.01*np.std(xdts), len(xdts))

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
MCMC = 4000  # No. of samples in Markov Chain,
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

for i in range(1, MCMC):
    print(i)
    
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

# Post processing:
zstorediff = zstore[:, burnin:]    
Zmeandiff = np.mean(zstorediff, axis=1)

thetadiff = theta[:, burnin:]
mutdiff = np.mean(thetadiff, axis=1)
sigtdiff = np.cov(thetadiff, bias = False)


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
