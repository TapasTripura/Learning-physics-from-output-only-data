"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). 
-- 'A sparse Bayesian framework for discovering interpretable nonlinear stochastic
    dynamical systems with Gaussian white noise'.
    (in Arxiv: 'Learning governing physics from output only measurements')
   
*** This code is for generation of the data.

# If you use any part of this code, then please cite us.

@article{tripura2022learning,
  title={Learning governing physics from output only measurements},
  author={Tripura, Tapas and Chakraborty, Souvik},
  journal={arXiv preprint arXiv:2208.05609},
  year={2022}
}

# The citation can also be found in Elsevier, MSSP journal.
"""

import numpy as np

"""
A Duffing Van der pol system excited by random noise
----------------------------------------------------------------------
"""
def duffing(x1, x2, T):
    # parameters of Duffing oscillator in Equation
    m = 1
    c = 2
    k = 1000
    k3 = 100000
    sigma = 10 # 1 (paper)
    
    # solution by Taylor 1.5 strong scheme Run with dt=0.01
    # -------------------------------------------------------
    np.random.seed(0)
    dt = 0.001
    t = np.arange(0, T+dt, dt)
    Nsamp = 200 # no. of samples in the run
    delmat = np.row_stack(([np.sqrt(dt), 0],[(dt**1.5)/2, (dt**1.5)/(2*np.sqrt(3))]))
    
    y1 = []
    y2 = []
    xz = []
    xzs = []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        x0 = np.array([x1, x2])
        x = np.vstack(x0)  # Zero initial condition.
        for n in range(len(t)-1):
            delgen = np.dot(delmat, np.random.normal(0,1,2))
            dW = delgen[0]
            dZ = delgen[1]
            a1 = x0[1]
            a2 = -(c/m)*x0[1]-(k/m)*x0[0]-(k3/m)*(x0[0]**3)
            b2 = (sigma/m)*x0[0]
            L0a1 = a2
            L0a2 = a1*(-(k/m)-(3*k3/m)*(x0[0]**2))+a2*(-(c/m))
            L0b2 = a1*(sigma/m)
            L1a1 = b2
            L1a2 = b2*(-(c/m))
            
            # Taylor 1.5 Mapping:
            # sol1 = x0[0] + a1*dt + 0.5*L0a1*(dt**2) + L1a1*dZ
            # sol2 = x0[1] + a2*dt + b2*dW + 0.5*L0a2*(dt**2) + L1a2*dZ + L0b2*(dW*dt-dZ)
            
            # Euler-Maruyama Mapping:
            sol1 = x0[0] + a1*dt 
            sol2 = x0[1] + a2*dt + b2*dW 
            
            x0 = np.array([sol1, sol2])
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        
        zint = x[1,0:-1]
        xfinal = x[1,1:] 
        xmz = (xfinal - zint) # 'x(t)-z' vector
        xmz2 = np.multiply(xmz, xmz)
        xz.append(xmz)
        xzs.append(xmz2)
        
    xz = pow(dt,-1)*np.mean(np.array(xz), axis = 0)
    xzs = pow(dt,-1)*np.mean(np.array(xzs), axis = 0)
    
    y1 = np.array(y1)
    y2 = np.array(y2)
    time = t[0:-1]
    
    return xz, xzs, y1, y2, time


"""
A Coulomb-Type base isolation system excited by random noise
---------------------------------------------------------------------
"""
def ddf(x):
    if x == 0:
        val = 1
    elif x != 0:
        val = 0
    return val


def coulomb(x1, x2, T):
    # parameters of Coulomb oscillator in Equation
    # ---------------------------------------------
    m = 1
    c = 2
    k = 3000
    mu = 1
    gf = 9.81
    sigma = 1
    
    # solution by Taylor 1.5 strong scheme Run with dt=0.01
    # -------------------------------------------------------
    dt = 0.001
    t = np.arange(0, T+dt, dt)
    Nsamp = 200 #int(1/dt) # no. of samples in the run
    
    delmat = np.row_stack(([np.sqrt(dt), 0],[(dt**1.5)/2, (dt**1.5)/(2*np.sqrt(3))]))
    
    y1 = []
    y2 = []
    xz = []
    xzs = []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        x0 = np.array([x1, x2])
        x = np.vstack(x0)  # Zero initial condition.
        for n in range(len(t)-1):
            delgen = np.dot(delmat, np.random.normal(0,1,2))
            dW = delgen[0]
            dZ = delgen[1]
            a1 = x0[1]
            a2 = -(c/m)*x0[1]-mu*gf*np.sign(x0[1])-(k/m)*x0[0]
            b1 = 0
            b2 = sigma/m
            L0a1 = a2
            L0a2 = a1*(-(k/m))+a2*(-(c/m)-mu*gf*2*ddf(x0[1]))
            L0b1 = 0
            L0b2 = a1/m
            L1a1 = b2
            L1a2 = b2*(-(c/m)-mu*gf*2*ddf(x0[1]))
            L1b1 = 0
            L1b2 = 0
            
            # Taylor 1.5 Mapping:
            # sol1 = x0[0] + a1*dt + 0.5*L0a1*(dt**2) + L1a1*dZ
            # sol2 = x0[1] + a2*dt + b2*dW + 0.5*L0a2*(dt**2) + L1a2*dZ + L0b2*(dW*dt-dZ)
            
            # Euler Maruyama Mapping:
            sol1 = x0[0] + a1*dt 
            sol2 = x0[1] + a2*dt + b2*dW 
            
            x0 = np.array([sol1, sol2])
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        
        zint = x[1,0:-1]
        xfinal = x[1,1:] 
        xmz = (xfinal - zint) # 'x(t)-z' vector
        xmz2 = np.multiply(xmz, xmz)
        xz.append(xmz)
        xzs.append(xmz2)
        
    xz = pow(dt,-1)*np.mean(np.array(xz), axis = 0)
    xzs = pow(dt,-1)*np.mean(np.array(xzs), axis = 0)
    
    y1 = np.array(y1)
    y2 = np.array(y2)
    time = t[0:-1]

    return xz, xzs, y1, y2, time


"""
Black-Scholes equation ::
-----------------------------------------------------------------
"""
def blackscholes(x1, T):
    # parameters of Black-Scholes equation:
    # ---------------------------------------------
    lam = 2   
    mu = 1
    cnst = lam-(mu**2)/2
    
    xzero = x1
    T = 1
    dt1 = 0.001
    t1 = np.arange(0, T+dt1, dt1)
    Nsamp = 500
    np.random.seed(0)
    delmat = np.row_stack(([np.sqrt(dt1), 0],[(dt1**1.5)/2, (dt1**1.5)/(2*np.sqrt(3))]))
    
    y = []
    xogen = []
    xz = []
    xzs = []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        # print(ensemble)
        xold = xzero
        xnew = xzero
        
        dB = np.sqrt(dt1)*np.random.randn(1)
        B = dB
        xtrue = xzero*np.exp(cnst*t1[0]+mu*B)
        
        for k in range(1, len(t1)):
            delgen = np.dot(delmat, np.random.randn(2))
            dB = delgen[0]
            dZ = delgen[1]
            a = lam*xold
            b = mu*xold
            L0a = a*lam
            L0b = a*mu
            L1a = b*lam
            L1b = b*mu
            L1L1b = b*(mu**2)
            
            # dB = np.sqrt(dt1)*np.random.randn(1)
            B = B + dB
            sol = xzero*np.exp(cnst*t1[k]+mu*B)
            xtrue = np.append(xtrue, sol)
            
            # sol = xold+ lam*xold*dt1 + mu*xold*dB
            sol = xold + a*dt1 + b*dB + 0.5*L0a*(dt1**2) + L1a*dZ + L1b*0.5*(dB**2-dt1) \
                + L0b*(dt1*dB-dZ) + 0.5*L1L1b*((dB**2)/3 -dt1)*dB
            xold = sol
            xnew = np.append(xnew, sol)
            
        y.append(xnew)
        xogen.append(xtrue)
    
        zint = xnew[0:-1]
        xfinal = xnew[1:] 
        xmz = (xfinal - zint) # 'x(t)-z' vector
        xmz2 = np.multiply(xmz, xmz)
        xz.append(xmz)
        xzs.append(xmz2)
        
    xz = pow(dt1,-1)*np.mean(np.array(xz), axis = 0)
    xzs = pow(dt1,-1)*np.mean(np.array(xzs), axis = 0)
    y = np.array(y)
    time = t1[0:-1]
    
    return xz, xzs, y, time    


"""
2-DOF system with top-linear and bottom base isolated ::
-----------------------------------------------------------------
"""
def dof2sys(x1, x2, x3, x4, T):
    # parameters of 2-DOF in Equation
    # ---------------------------------------------
    m1, m2 = 1, 1
    k1, k2 = 4000, 2000
    c1, c2 = 2, 2
    mu, gf = 1, 9.81
    sigma1, sigma2 = 10, 10
    
    # solution by Taylor 1.5 strong scheme Run with dt=0.01
    # -------------------------------------------------------
    T = 1
    dt = 0.001
    t = np.arange(0, T+dt, dt)
    Nsamp = 200 #int(1/dt) # no. of samples in the run
    
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    xz1 = []
    xz2 = []
    xzs11 = []
    xzs12 = []
    xzs22 = []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        x0 = np.array([x1, x2, x3, x4])
        x = x0  # Zero initial condition.
        for n in range(len(t)-1):
            dW = np.sqrt(dt)*np.random.randn(2)
            dW1 = dW[0]
            dW2 = dW[1]
            
            a1 = x0[1]
            a2 = -(c1/m1)*x0[1]-mu*gf*np.sign(x0[1])-(k1/m1)*x0[0] \
                -(c2/m1)*(x0[1]-x0[3])-(k2/m1)*(x0[0]-x0[2])
            a3 = x0[3]
            a4 = -(c2/m2)*(x0[3]-x0[1])-(k2/m2)*(x0[2]-x0[0])
            b1 = 0
            b2 = (sigma1)/m1
            b3 = 0
            b4 = (sigma2)/m2
            
            sol1 = x0[0] + a1*dt
            sol2 = x0[1] + a2*dt + b2*dW1
            sol3 = x0[2] + a3*dt
            sol4 = x0[3] + a4*dt + b4*dW2
            
            x0 = np.array([sol1, sol2, sol3, sol4])
            x = np.column_stack((x, x0))
            
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
        y4.append(x[3,:])
        
        zint1 = x[1, 0:-1]
        xfinal1 = x[1, 1:] 
        xmz1 = (xfinal1 - zint1) # 'x1(t)-z1' vector
        zint2 = x[3, 0:-1]
        xfinal2 = x[3, 1:] 
        xmz2 = (xfinal2 - zint2) # 'x2(t)-z2' vector
        
        xz1.append(xmz1)
        xz2.append(xmz2)
        
        xmzsq11 = np.multiply(xmz1, xmz1)
        xzs11.append(xmzsq11)
        
        xmzsq12 = np.multiply(xmz1, xmz2)
        xzs12.append(xmzsq12)
        
        xmzsq22 = np.multiply(xmz2, xmz2)
        xzs22.append(xmzsq22)
        
    xz1 = pow(dt,-1)*np.mean(np.array(xz1), axis = 0)
    xz2 = pow(dt,-1)*np.mean(np.array(xz2), axis = 0)
    
    xzs11 = pow(dt,-1)*np.mean(np.array(xzs11), axis = 0)
    xzs12 = pow(dt,-1)*np.mean(np.array(xzs12), axis = 0)
    xzs22 = pow(dt,-1)*np.mean(np.array(xzs22), axis = 0)
    
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    y4 = np.array(y4)
    time = t[0:-1]
    
    return xz1, xz2, xzs11, xzs12, xzs22, y1, y2, y3, y4, time


"""
A Bouc-Wen oscillator excited by random noise
--> Partially oberved variable Z(t)
--------------------------------------------------------------------------
"""
def boucwen(x1, x2, x3, T):
    # parameters of Bouc-Wen oscillator in Equation
    # ---------------------------------------------
    m = 1
    c = 20
    k = 10000
    lam = 0.5
    A1 = 0.5
    A2 = 0.5
    A3 = 1
    nbar = 3
    sigma1 = 2
    
    # solution by Taylor 1.5 strong scheme Run with dt=0.01
    # -------------------------------------------------------
    dt = 0.001
    t = np.arange(0, T+dt, dt)
    Nsamp = 200 #int(1/dt) # no. of samples in the run
    
    np.random.seed(2021)
    force = np.random.randn(len(t), Nsamp)
    
    y1 = []
    y2 = []
    y3 = []
    xz1 = []
    xzs11 = []
    # Simulation Starts Here ::
    # -------------------------------------------------------
    for ensemble in range(Nsamp):
        x0 = np.array([x1, x2, x3])
        x = np.vstack(x0)  # Zero initial condition.
        for n in range(len(t)-1):
            a1 = x0[1]
            a2 = -(c/m)*x0[1]-(k/m)*lam*x0[0]-(k/m)*(1-lam)*x0[2]
            a3 = -A1*x0[2]*np.abs(x0[1])*pow(np.abs(x0[2]),nbar-1) \
                -A2*x0[1]*pow(np.abs(x0[2]),nbar) + A3*x0[2]
            b1 = 0
            b2 = (sigma1/m)
            b3 = 0
    
            sol1 = x0[0] + a1*dt 
            sol2 = x0[1] + a2*dt + b2*force[n,ensemble]*np.sqrt(dt)
            sol3 = x0[2] + a3*dt 
            x0 = np.array([sol1, sol2, sol3])
            x = np.column_stack((x, x0))
        y1.append(x[0,:])
        y2.append(x[1,:])
        y3.append(x[2,:])
        
        zint1 = x[0, 0:-1]
        xfinal1 = x[0, 1:] 
        xmz1 = (xfinal1 - zint1) # 'x1(t)-z1' vector
        
        xz1.append(xmz1)        
        xmzsq11 = np.multiply(xmz1, xmz1)
        xzs11.append(xmzsq11)
        
    xz1 = pow(dt,-1)*np.mean(np.array(xz1), axis = 0)    
    xzs11 = pow(dt,-1)*np.mean(np.array(xzs11), axis = 0)
    
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    time = t[0:-1]
    
    # return xz1, xzs11, y1, y2, y3, time   # when hysteresis z(t) is known,
    return xz1, xzs11, y1, y2, time   # hysteresis z(t) is unkown,
