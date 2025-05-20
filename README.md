import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# 1. Define Constants and Functions
# ==============================
Delta = 992887573 / 5367303  # Precomputed constant

def Theta1(x):
    """First component of the kernel"""
    return (1850/207) * (x ** (23/25)) + (242/43)

def Theta2(x):
    """Second component of the kernel"""
    return (48850/4623) * (1 - (x ** (23/25))) + (95/18)

# ==============================
# 2. Create Computation Grid
# ==============================
N = 150  # Grid resolution
rho = np.linspace(0, 1, N)
xi = np.linspace(0, 1, N)
Rho, Xi = np.meshgrid(rho, xi)

# ==============================
# 3. Compute Green's Function
# ==============================
aleph = np.zeros_like(Rho)

# Case 1: ξ ≤ ϱ
mask_lower = Xi <= Rho
aleph[mask_lower] = (Theta1(Xi[mask_lower]) * Theta2(Rho[mask_lower])) / Delta

# Case 2: ϱ ≤ ξ
mask_upper = Rho < Xi
aleph[mask_upper] = (Theta1(Rho[mask_upper]) * Theta2(Xi[mask_upper])) / Delta

# ==============================
# 4. Create 3D Surface Plot
# ==============================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(Rho, Xi, aleph, cmap='viridis', 
                      rstride=2, cstride=2, 
                      linewidth=0.1, antialiased=True)

ax.set_xlabel(r'$\rho$', fontsize=14, labelpad=12)
ax.set_ylabel(r'$\xi$', fontsize=14, labelpad=12)
ax.set_zlabel(r'$\aleph(\rho, \xi)$', fontsize=14, labelpad=12)
ax.set_title("Green's Function Surface Plot", fontsize=16, pad=20)

# Customize view angle
ax.view_init(elev=25, azim=-120)

# Add color bar
cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label(r'$\aleph(\rho, \xi)$', fontsize=12)

plt.tight_layout()

# ==============================
# 5. Create Contour Plot
# ==============================
plt.figure(figsize=(10, 8))
cp = plt.contourf(Rho, Xi, aleph, levels=50, cmap='viridis')

plt.xlabel(r'$\rho$', fontsize=14)
plt.ylabel(r'$\xi$', fontsize=14)
plt.title("Green's Function Contour Plot", fontsize=16)

cbar = plt.colorbar(cp)
cbar.set_label(r'$\aleph(\rho, \xi)$', fontsize=12)

plt.grid(linestyle='--', alpha=0.4)
plt.tight_layout()

# ==============================
# 6. Show Plots
# ==============================
plt.show()
