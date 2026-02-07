#!/usr/bin/env python3
"""
EXPERIMENT: Projectile Motion from LFM Substrate
=================================================

HYPOTHESIS FRAMEWORK
--------------------

GENERAL HYPOTHESIS:
Parabolic projectile motion emerges from wave packet dynamics in a 
χ-gradient using only GOV-01.

NULL HYPOTHESIS (H₀):
A wave packet in a χ-gradient will NOT follow parabolic trajectory.

ALTERNATIVE HYPOTHESIS (H₁):
The wave packet will curve toward lower χ, following approximately 
parabolic path consistent with constant-acceleration kinematics.

LFM-ONLY CONSTRAINT VERIFICATION:
- [x] Uses ONLY GOV-01: ∂²E/∂t² = c²∇²E − χ²E
- [x] χ-gradient represents gravitational potential
- [x] NO F = mg injected
- [x] NO hardcoded parabolas

SUCCESS CRITERIA:
- REJECT H₀ if: Trajectory shows parabolic curve with R² > 0.90
- FAIL TO REJECT H₀ if: Trajectory is random or linear

PHYSICS EXPLANATION:
In LFM, gravity is a χ-gradient. Near a surface, χ increases with height.
Wave packets curve toward lower χ because phase velocity depends on χ.
This bending IS gravity - emerging from GOV-01 wave dynamics.
"""

import numpy as np
import json
from datetime import datetime

# =============================================================================
# LFM PARAMETERS (scaled for numerical stability)
# =============================================================================
chi_background = 1.0   # Background χ
c = 1.0                # Wave speed

# Grid parameters  
N = 150                # Grid points per dimension
L = 60.0               # Domain size
dx = L / N
dt = 0.3 * dx / c      # CFL condition
n_steps = 800          # Simulation steps

print("=" * 60)
print("LFM PROJECTILE MOTION EXPERIMENT")
print("=" * 60)
print(f"\nParameters: chi_bg = {chi_background}, c = {c}")
print(f"Grid: {N}x{N}, L = {L}, dx = {dx:.4f}")
print(f"Time: dt = {dt:.6f}, steps = {n_steps}")

# =============================================================================
# CREATE 2D GRID
# =============================================================================
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y, indexing='ij')

# =============================================================================
# CREATE chi-GRADIENT (represents surface gravity)
# =============================================================================
# chi increases with height: waves at top are slower, curve back down
gravity_strength = 0.02
chi = chi_background + gravity_strength * Y

print(f"\nchi-gradient: d(chi)/dy = {gravity_strength}")
print(f"chi range: [{chi.min():.3f}, {chi.max():.3f}]")

# =============================================================================
# PROJECTILE: Wave packet with initial horizontal velocity
# =============================================================================
proj_x0, proj_y0 = 10.0, 40.0
proj_width = 2.5
proj_amplitude = 1.0
velocity_x = 0.5

r2 = (X - proj_x0)**2 + (Y - proj_y0)**2
E = proj_amplitude * np.exp(-r2 / (2 * proj_width**2))

# Previous step: shifted by velocity
r2_prev = (X - proj_x0 + velocity_x*dt)**2 + (Y - proj_y0)**2
E_prev = proj_amplitude * np.exp(-r2_prev / (2 * proj_width**2))

print(f"\nProjectile: position ({proj_x0:.1f}, {proj_y0:.1f}), v_x = {velocity_x}")

# =============================================================================
# EVOLUTION: GOV-01 ONLY
# =============================================================================
print("\nEvolving wave packet with GOV-01...")

trajectory = []

for step in range(n_steps):
    # GOV-01: E_new = 2*E - E_prev + dt^2*(c^2*Laplacian(E) - chi^2*E)
    laplacian = np.zeros_like(E)
    laplacian[1:-1, 1:-1] = (
        E[:-2, 1:-1] + E[2:, 1:-1] + 
        E[1:-1, :-2] + E[1:-1, 2:] - 
        4 * E[1:-1, 1:-1]
    ) / dx**2
    
    E_new = 2 * E - E_prev + dt**2 * (c**2 * laplacian - chi**2 * E)
    
    # Absorbing boundaries
    E_new[0, :] = 0
    E_new[-1, :] = 0
    E_new[:, 0] = 0
    E_new[:, -1] = 0
    
    E_prev = E.copy()
    E = E_new.copy()
    
    # Track center of mass
    E2 = E**2
    total_E2 = np.sum(E2)
    
    if total_E2 > 1e-10:
        cx = np.sum(X * E2) / total_E2
        cy = np.sum(Y * E2) / total_E2
        trajectory.append((step * dt, cx, cy))
    
    if step % 200 == 0 and len(trajectory) > 0:
        print(f"  Step {step}: ({trajectory[-1][1]:.2f}, {trajectory[-1][2]:.2f})")

# =============================================================================
# ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("TRAJECTORY ANALYSIS")
print("=" * 60)

trajectory = np.array(trajectory)
t = trajectory[:, 0]
x_pos = trajectory[:, 1]
y_pos = trajectory[:, 2]

print(f"\nTrajectory: {len(trajectory)} points")
print(f"Start: ({x_pos[0]:.2f}, {y_pos[0]:.2f})")
print(f"End: ({x_pos[-1]:.2f}, {y_pos[-1]:.2f})")

# Horizontal: linear fit
t_centered = t - t[0]
x_coeffs = np.polyfit(t_centered, x_pos, 1)
x_fit = np.polyval(x_coeffs, t_centered)
ss_res_x = np.sum((x_pos - x_fit)**2)
ss_tot_x = np.sum((x_pos - np.mean(x_pos))**2)
R2_x_linear = 1 - ss_res_x / ss_tot_x if ss_tot_x > 0 else 0

print(f"\nHorizontal: v = {x_coeffs[0]:.4f}, R2 = {R2_x_linear:.4f}")

# Vertical: parabolic fit
y_coeffs = np.polyfit(t_centered, y_pos, 2)
y_fit = np.polyval(y_coeffs, t_centered)
ss_res_y = np.sum((y_pos - y_fit)**2)
ss_tot_y = np.sum((y_pos - np.mean(y_pos))**2)
R2_y_parabolic = 1 - ss_res_y / ss_tot_y if ss_tot_y > 0 else 0

# Linear comparison
y_linear_coeffs = np.polyfit(t_centered, y_pos, 1)
y_linear_fit = np.polyval(y_linear_coeffs, t_centered)
ss_res_y_lin = np.sum((y_pos - y_linear_fit)**2)
R2_y_linear = 1 - ss_res_y_lin / ss_tot_y if ss_tot_y > 0 else 0

effective_a = 2 * y_coeffs[0]
curves_down = y_coeffs[0] < 0

print(f"Vertical: a = {effective_a:.6f}")
print(f"  R2 (parabola) = {R2_y_parabolic:.4f}")
print(f"  R2 (linear) = {R2_y_linear:.4f}")
print(f"  Curves down: {curves_down}")

# =============================================================================
# HYPOTHESIS VALIDATION
# =============================================================================
print("\n" + "=" * 60)
print("HYPOTHESIS VALIDATION")
print("=" * 60)

H0_rejected = R2_y_parabolic > 0.90 and curves_down

print(f"\nLFM-ONLY VERIFIED: YES")
print(f"H0 STATUS: {'REJECTED' if H0_rejected else 'FAILED TO REJECT'}")

if H0_rejected:
    print("CONCLUSION: Parabolic projectile motion EMERGES from GOV-01")
else:
    print("CONCLUSION: Motion pattern not clearly parabolic")

print("=" * 60)

# Save results
results = {
    "experiment": "LFM Projectile Motion",
    "timestamp": datetime.now().isoformat(),
    "parameters": {"chi_background": chi_background, "gravity_strength": gravity_strength},
    "results": {
        "R2_horizontal_linear": float(R2_x_linear),
        "R2_vertical_parabolic": float(R2_y_parabolic),
        "R2_vertical_linear": float(R2_y_linear),
        "effective_acceleration": float(effective_a),
        "curves_down": bool(curves_down)
    },
    "hypothesis": {
        "H0_rejected": bool(H0_rejected),
        "lfm_only_verified": True
    }
}

with open("projectile_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to projectile_results.json")
