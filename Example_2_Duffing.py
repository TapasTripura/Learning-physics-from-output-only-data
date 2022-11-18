"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). 
-- 'A sparse Bayesian framework for discovering interpretable nonlinear stochastic
    dynamical systems with Gaussian white noise'.
    (in Arxiv: 'Learning governing physics from output only measurements')
   
*** This code is for the Example 2: Parametrically excited Duffing oscillator.

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
import seaborn as sns

# %%
"""
For the drift identification :::
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Response generation:
T = 1
x1, x2 = 0.1, 0 # initial displacement for duffing
xdt, bxt, y1, y2, t_eval = utils_response.duffing(x1, x2, T)

# Expected Dictionary Creation:
libr = []
for j in range(len(y1)):
    data = np.row_stack((y1[j,0:-1], y2[j,0:-1]))
    Dtemp, nl = utils_library.library(data, polyn=6, modulus=1, harmonic=0)
    libr.append(Dtemp)
libr = np.array(libr)
D= np.mean(libr, axis = 0)

xdts = xdt

# Adding noise:
noise = 0.01
xdts = xdts + np.random.normal(0, noise*np.std(xdts), len(xdts))

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
    if i % 50 == 0:
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
x1, x2 = 0.005, 0 # initial displacement condition
gg, xbt, y1, y2, t_eval = utils_response.duffing(x1, x2, T)

# Dictionary Creation:
libr = []
for j in range(len(y1)):
    data = np.row_stack((y1[j,0:-1], y2[j,0:-1]))
    Dtemp, nl = utils_library.library(data, polyn=6, modulus=0, harmonic=0)
    libr.append(Dtemp)
libr = np.array(libr)
D = np.mean(libr, axis = 0)

xdts = xbt

# Adding noise:
xdts = xdts + np.random.normal(0, noise*np.std(xdts), len(xdts))

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
    if i % 50 == 0:
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

zstorediff = zstore[:, burnin:]    
Zmeandiff = np.mean(zstorediff, axis=1)

thetadiff = theta[:, burnin:]
mutdiff = np.mean(thetadiff, axis=1)
sigtdiff = np.cov(thetadiff, bias = False)

# Post processing:
less_uncertain = 0.02
mutind = np.where(mutdiff<(np.max(np.abs(mutdiff))*less_uncertain))
Zmeandiff[mutind] = Zmeandiff[mutind]*0
thetanor = thetadiff[3,:] - np.abs(thetadrift[1,:])
munormal = np.mean(thetanor)

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
