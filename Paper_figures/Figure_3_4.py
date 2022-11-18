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

import numpy as np
import matplotlib.pyplot as plt

# %%
import pickle
[ZdriftBS,ZdiffBS,ZdriftBSW,ZdiffBSW,ZdriftDVP,ZdiffDVP,Zdrift12dof,Zdrift22dof, \
 Zdiff12dof,Zdiff22dof] = pickle.load(open("actual_data/data_stem.p","rb"))

# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18

# Plot for the case with -reduced library functions,
figure1=plt.figure(figsize = (16, 12))
plt.subplot(511)
xr = np.array(range(len(ZdriftBS)))
plt.stem(xr, ZdriftBS, use_line_collection = True, linefmt='blue', basefmt="k", markerfmt='bo')
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=3)
plt.ylabel('PIP (drift)', fontweight='bold');
plt.title('(a) Black-Scholes SDE (library with |x| and x|x|)', fontweight='bold')
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.text(1, 0.7, "X", color='b', fontsize=18, fontweight='bold')
plt.text(8, 0.7, "$|$X$|$", color='b', fontsize=18, fontweight='bold')
plt.text(1, -0.2, "$1$", fontsize=18, fontweight='bold')

plt.subplot(512)
xr = np.array(range(len(ZdriftBSW)))
plt.stem(xr, ZdriftBSW, use_line_collection = True, linefmt='blue', basefmt="k", markerfmt='bo')
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=3)
plt.ylabel('PIP (drift)', fontweight='bold');
plt.title('(b) Black-Scholes SDE (library without |x| and x|x|)', fontweight='bold')
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.text(1, 1.1, "X", color='b', fontsize=18, fontweight='bold')

plt.subplot(513)
xr = np.array(range(len(ZdriftDVP)))
plt.stem(xr, ZdriftDVP, use_line_collection = True, linefmt='blue', basefmt="k", markerfmt='bD')
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=3)
plt.ylabel('PIP (drift)', fontweight='bold');
plt.title('(c) Duffing oscillator with parametric excitation', fontweight='bold')
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.text(1.1, 1.1, "X$_1$", color='b', fontsize=18, ha='right', fontweight='bold')
plt.text(2.1, 1.1, "X$_2$", color='b', fontsize=18, fontweight='bold')
plt.text(6.1, 1.1, "X$_1^3$", color='b', fontsize=18, fontweight='bold')
plt.text(1, -0.2, "$1$", fontsize=18, fontweight='bold')
plt.text(2, -0.2, "$2$", fontsize=18, fontweight='bold')
plt.text(6, -0.2, "$6$", fontsize=18, fontweight='bold')

plt.subplot(514)
xr = np.array(range(len(Zdrift12dof)))
plt.stem(xr, Zdrift12dof, use_line_collection = True, linefmt='blue', basefmt="k", markerfmt='bs')
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=3)
plt.ylabel('PIP (drift)', fontweight='bold');
plt.title('(d) Two-DOF shear building frame with Coulomb base isolator (first-DOF)', fontweight='bold')
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.text(1.1, 1.1, "X$_1$", color='b', fontsize=18, ha='right', fontweight='bold')
plt.text(1.3, 1.1, "X$_2$", color='b', fontsize=18, fontweight='bold')
plt.text(2.7, 1.1, "X$_3$", color='b', fontsize=18, fontweight='bold')
plt.text(4.1, 1.1, "X$_4$", color='b', fontsize=18, fontweight='bold')
plt.text(36.3, 0.8, "sgn(X$_1$)", color='b', fontsize=20, fontweight='bold')
plt.text(1, -0.2, "$1$", fontsize=18, fontweight='bold')
plt.text(2, -0.2, "$2$", fontsize=18, fontweight='bold')
plt.text(3, -0.2, "$3$", fontsize=18, fontweight='bold')
plt.text(4, -0.2, "$4$", fontsize=18, fontweight='bold')
plt.text(36, -0.2, "$36$", fontsize=18, fontweight='bold')

plt.subplot(515)
xr = np.array(range(len(Zdrift22dof)))
plt.stem(xr, Zdrift22dof, use_line_collection = True, linefmt='blue', basefmt="k", markerfmt='bs')
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=3)
plt.xlabel('Library functions', fontweight='bold');
plt.ylabel('PIP (drift)', fontweight='bold');
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.text(1.1, 1.1, "X$_1$", color='b', fontsize=18, ha='right', fontweight='bold')
plt.text(1.3, 1.1, "X$_2$", color='b', fontsize=18, fontweight='bold')
plt.text(2.7, 1.1, "X$_3$", color='b', fontsize=18, fontweight='bold')
plt.text(4.1, 1.1, "X$_4$", color='b', fontsize=18, fontweight='bold')
plt.text(1, -0.2, "$1$", fontsize=18, fontweight='bold')
plt.text(2, -0.2, "$2$", fontsize=18, fontweight='bold')
plt.text(3, -0.2, "$3$", fontsize=18, fontweight='bold')
plt.text(4, -0.2, "$4$", fontsize=18, fontweight='bold')
plt.title('(e) Two-DOF shear building frame with Coulomb base isolator (second-DOF)', fontweight='bold')
plt.tight_layout()
plt.show()

# %%
figure2=plt.figure(figsize = (16, 12))
plt.subplot(511)
xr = np.array(range(len(ZdiffBS)))
plt.stem(xr, ZdiffBS, use_line_collection = True, linefmt='blue', basefmt="k", markerfmt='bo')
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=3)
plt.ylabel('PIP (diffusion)', fontweight='bold');
plt.title('(a) Black-Scholes SDE (library with |x| and x|x|)', fontweight='bold')
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.text(2, 0.7, "X$^2$", color='b', fontsize=18, fontweight='bold')
plt.text(9, 0.7, "X$|X|$", color='b', fontsize=18, fontweight='bold')
plt.text(9, -0.2, "$9$", fontsize=18, fontweight='bold')

plt.subplot(512)
xr = np.array(range(len(ZdiffBSW)))
plt.stem(xr, ZdiffBSW, use_line_collection = True, linefmt='blue', basefmt="k", markerfmt='bo')
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=3)
plt.ylabel('PIP (diffusion)', fontweight='bold');
plt.title('(b) Black-Scholes SDE (library without |x| and x|x|)', fontweight='bold')
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.text(2.1, 0.8, "X$^2$", color='b', fontsize=18, fontweight='bold')

plt.subplot(513)
xr = np.array(range(len(ZdiffDVP)))
plt.stem(xr, ZdiffDVP, use_line_collection = True, linefmt='blue', basefmt="k", markerfmt='bD')
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=3)
plt.ylabel('PIP (diffusion)', fontweight='bold');
plt.title('(c) Duffing oscillator with parametric excitation', fontweight='bold')
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.text(3.3, 0.8, "X$_1^2$", color='b', fontsize=18, fontweight='bold')
plt.text(3, -0.2, "$3$", fontsize=18, fontweight='bold')

plt.subplot(514)
xr = np.array(range(len(Zdiff12dof)))
plt.stem(xr, Zdiff12dof, use_line_collection = True, linefmt='blue', basefmt="k", markerfmt='bs')
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=3)
plt.ylabel('PIP (diffusion)', fontweight='bold');
plt.title('(d) Two-DOF shear building frame with Coulomb base isolator (first-DOF)', fontweight='bold')
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.text(0.5, 0.8, "$1$", color='b', fontsize=18, fontweight='bold')

plt.subplot(515)
xr = np.array(range(len(Zdiff22dof)))
plt.stem(xr, Zdiff22dof, use_line_collection = True, linefmt='blue', basefmt="k", markerfmt='bs')
plt.axhline(y= 0.5, color='r', linestyle='-.', linewidth=3)
plt.xlabel('Library functions', fontweight='bold');
plt.ylabel('PIP (diffusion)', fontweight='bold');
plt.ylim(0,1.05)
plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
plt.text(0.5, 0.8, "$1$", color='b', fontsize=18, fontweight='bold')
plt.title('(e) Two-DOF shear building frame with Coulomb base isolator (second-DOF)', fontweight='bold')

plt.tight_layout()
plt.show()
