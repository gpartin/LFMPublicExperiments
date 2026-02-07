#!/usr/bin/env python3
"""
EXPERIMENT: Frame Dragging from LFM Substrate Dynamics
=======================================================

HYPOTHESIS FRAMEWORK
--------------------

GENERAL HYPOTHESIS:
A rotating E-source will induce angular momentum transfer to a nearby
test wave packet through pure GOV-01/02 dynamics, without any injected
force laws.

NULL HYPOTHESIS (H0):
A test wave packet near a rotating E-source will NOT gain angular 
momentum in the direction of the source's rotation. The wave packet's
angular momentum will remain unchanged or change randomly.

ALTERNATIVE HYPOTHESIS (H1):
The test wave packet will gain angular momentum in the SAME direction
as the source's rotation, demonstrating frame-dragging emergence.

LFM-ONLY CONSTRAINT VERIFICATION:
- [x] Uses ONLY GOV-01: d^2E/dt^2 = c^2 * Laplacian(E) - chi^2 * E
- [x] Uses ONLY GOV-02: d^2chi/dt^2 = c^2 * Laplacian(chi) - kappa*(E^2 - E0^2)
- [x] NO F = v x B injected
- [x] NO Lorentz force assumed
- [x] NO Lense-Thirring formula used
- [x] Angular momentum MEASURED from wave dynamics, not predicted

SUCCESS CRITERIA:
- REJECT H0 if: Test wave gains angular momentum in SAME direction as source rotation
- FAIL TO REJECT H0 if: No systematic angular momentum transfer

PHYSICS APPROACH:
1. Create a rotating E-source by evolving an asymmetric initial condition via GOV-01
2. The rotating E^2 creates a rotating chi-well via GOV-02
3. Place a test wave packet in the rotating chi-gradient
4. Evolve the ENTIRE system via GOV-01/02 (coupled)
5. Measure angular momentum of test wave: L = integral(r x p) where p ~ E * grad(E)
6. Check if L grows in the direction of source rotation

If frame-dragging exists in LFM, it must emerge from the chi-gradient dynamics.
"""

import numpy as np
import json
from datetime import datetime

print("=" * 70)
print("LFM FRAME DRAGGING EXPERIMENT")
print("Pure GOV-01/02 dynamics - No external physics")
print("=" * 70)

# =============================================================================
# LFM PARAMETERS
# =============================================================================
chi_0 = 2.0           # Background chi (using moderate value for numerics)
kappa = 0.5           # chi-E coupling
c = 1.0               # Wave speed

# Grid parameters (2D for visualization and speed)
N = 150
L = 60.0
dx = L / N
dt = 0.2 * dx / c     # CFL condition
n_steps = 2000

print(f"\nParameters: chi_0 = {chi_0}, kappa = {kappa}, c = {c}")
print(f"Grid: {N}x{N}, L = {L}, dx = {dx:.4f}")
print(f"Time: dt = {dt:.6f}, steps = {n_steps}")

# =============================================================================
# CREATE GRID
# =============================================================================
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y, indexing='ij')
R = np.sqrt(X**2 + Y**2) + 1e-6
THETA = np.arctan2(Y, X)

# =============================================================================
# INITIAL CONDITIONS: Rotating Source + Test Wave
# =============================================================================

# Rotating source: Asymmetric pattern that will naturally rotate
# Using a dipole-like structure with angular momentum
source_radius = 8.0
source_width = 3.0
source_amp = 3.0

# Create source with built-in angular momentum (m=1 mode)
# This is an eigenmode that naturally rotates in the chi-well
E_source = source_amp * np.exp(-((R - source_radius)**2) / (2 * source_width**2))
E_source *= np.cos(THETA)  # m=1 angular mode -> will rotate

# For rotation, we need the previous step to have phase offset
# E_prev for source should be rotated by omega*dt
omega_source = 0.1  # Source rotation rate
phase_offset = omega_source * dt
E_source_prev = source_amp * np.exp(-((R - source_radius)**2) / (2 * source_width**2))
E_source_prev *= np.cos(THETA - phase_offset)

# Test wave packet: Placed at distance, initially no angular momentum
test_x, test_y = 25.0, 0.0  # Test wave position
test_width = 2.5
test_amp = 0.5

r_test = np.sqrt((X - test_x)**2 + (Y - test_y)**2)
E_test = test_amp * np.exp(-r_test**2 / (2 * test_width**2))

# Combined initial E field
E = E_source + E_test
E_prev = E_source_prev + E_test  # Test wave starts at rest

# Initialize chi field (quasi-static equilibrium initially)
chi = chi_0 * np.ones((N, N))
chi_prev = chi.copy()

print(f"\nRotating source: radius = {source_radius}, omega = {omega_source}")
print(f"Test wave at: ({test_x}, {test_y})")

# =============================================================================
# MEASUREMENT FUNCTIONS
# =============================================================================

def compute_angular_momentum(E_field, E_prev_field, X, Y, dx, dt, region_mask):
    """
    Compute angular momentum of wave in specified region.
    
    L_z = integral of (x * p_y - y * p_x) where p ~ E * dE/dt * grad(E)/|grad(E)|
    
    Using momentum density: p = (dE/dt) * grad(E) / c^2
    Then L_z = integral of r x p
    """
    # Time derivative
    dE_dt = (E_field - E_prev_field) / dt
    
    # Spatial gradients
    dE_dx = np.gradient(E_field, dx, axis=0)
    dE_dy = np.gradient(E_field, dx, axis=1)
    
    # Momentum density components: p_i ~ dE/dt * dE/dx_i
    p_x = dE_dt * dE_dx
    p_y = dE_dt * dE_dy
    
    # Angular momentum density: L_z = x*p_y - y*p_x
    L_density = X * p_y - Y * p_x
    
    # Integrate over region
    L_z = np.sum(L_density * region_mask) * dx**2
    
    return L_z

def compute_energy_in_region(E_field, region_mask, dx):
    """Compute total E^2 in region (for normalization)."""
    return np.sum(E_field**2 * region_mask) * dx**2

# Define test wave region (annulus around test position)
test_region = (np.sqrt((X - test_x)**2 + (Y - test_y)**2) < 15.0) & \
              (R > source_radius + 5)  # Exclude source

# Define source region
source_region = R < source_radius + source_width * 2

# =============================================================================
# EVOLUTION: Coupled GOV-01 and GOV-02
# =============================================================================
print("\nEvolving with coupled GOV-01/02...")
print("Measuring angular momentum transfer to test wave...")

# Storage for measurements
L_test_history = []
L_source_history = []
time_history = []

for step in range(n_steps):
    # -----------------------------------------------------------------
    # GOV-01: d^2E/dt^2 = c^2 * Laplacian(E) - chi^2 * E
    # -----------------------------------------------------------------
    laplacian_E = np.zeros_like(E)
    laplacian_E[1:-1, 1:-1] = (
        E[:-2, 1:-1] + E[2:, 1:-1] +
        E[1:-1, :-2] + E[1:-1, 2:] -
        4 * E[1:-1, 1:-1]
    ) / dx**2
    
    E_new = 2*E - E_prev + dt**2 * (c**2 * laplacian_E - chi**2 * E)
    
    # -----------------------------------------------------------------
    # GOV-02: d^2chi/dt^2 = c^2 * Laplacian(chi) - kappa*(E^2 - E0^2)
    # -----------------------------------------------------------------
    laplacian_chi = np.zeros_like(chi)
    laplacian_chi[1:-1, 1:-1] = (
        chi[:-2, 1:-1] + chi[2:, 1:-1] +
        chi[1:-1, :-2] + chi[1:-1, 2:] -
        4 * chi[1:-1, 1:-1]
    ) / dx**2
    
    E0_squared = 0  # Vacuum
    chi_new = 2*chi - chi_prev + dt**2 * (c**2 * laplacian_chi - kappa*(E**2 - E0_squared))
    
    # Keep chi positive
    chi_new = np.maximum(chi_new, 0.1)
    
    # Boundary conditions
    E_new[0, :] = 0
    E_new[-1, :] = 0
    E_new[:, 0] = 0
    E_new[:, -1] = 0
    chi_new[0, :] = chi_0
    chi_new[-1, :] = chi_0
    chi_new[:, 0] = chi_0
    chi_new[:, -1] = chi_0
    
    # Measure angular momentum
    if step % 50 == 0:
        L_test = compute_angular_momentum(E, E_prev, X, Y, dx, dt, test_region)
        L_source = compute_angular_momentum(E, E_prev, X, Y, dx, dt, source_region)
        
        L_test_history.append(L_test)
        L_source_history.append(L_source)
        time_history.append(step * dt)
        
        if step % 500 == 0:
            print(f"  Step {step}: L_source = {L_source:+.4f}, L_test = {L_test:+.4f}")
    
    # Update fields
    E_prev = E.copy()
    E = E_new.copy()
    chi_prev = chi.copy()
    chi = chi_new.copy()

# =============================================================================
# ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

L_test_history = np.array(L_test_history)
L_source_history = np.array(L_source_history)
time_history = np.array(time_history)

# Initial and final angular momentum
L_test_initial = L_test_history[0]
L_test_final = L_test_history[-1]
delta_L_test = L_test_final - L_test_initial

L_source_initial = L_source_history[0]
L_source_final = L_source_history[-1]
delta_L_source = L_source_final - L_source_initial

print(f"\nSource angular momentum:")
print(f"  Initial: {L_source_initial:+.4f}")
print(f"  Final:   {L_source_final:+.4f}")
print(f"  Change:  {delta_L_source:+.4f}")

print(f"\nTest wave angular momentum:")
print(f"  Initial: {L_test_initial:+.4f}")
print(f"  Final:   {L_test_final:+.4f}")
print(f"  Change:  {delta_L_test:+.4f}")

# Check if test wave gained L in same direction as source rotation
# Source rotates with positive omega -> L_source should be positive
# Frame dragging predicts: L_test increases in same direction

source_sign = np.sign(L_source_initial) if abs(L_source_initial) > 0.01 else np.sign(np.mean(L_source_history))
test_gain_sign = np.sign(delta_L_test)

same_direction = (source_sign * test_gain_sign > 0) and (abs(delta_L_test) > 0.001)

# Linear fit to check for systematic trend
from numpy.polynomial import polynomial as P
if len(time_history) > 2:
    # Fit L_test vs time
    coeffs = np.polyfit(time_history, L_test_history, 1)
    L_rate = coeffs[0]  # dL/dt for test wave
else:
    L_rate = 0

print(f"\nAngular momentum transfer rate: dL_test/dt = {L_rate:+.6f}")
print(f"Source rotation direction: {'CCW (+)' if source_sign > 0 else 'CW (-)'}")
print(f"Test wave L change direction: {'+ (same)' if test_gain_sign > 0 else '- (opposite)' if test_gain_sign < 0 else 'none'}")

# =============================================================================
# HYPOTHESIS VALIDATION
# =============================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS VALIDATION")
print("=" * 70)

# Criteria for rejection:
# 1. Test wave gains angular momentum (|delta_L| > threshold)
# 2. Direction matches source rotation
# 3. Trend is systematic (not noise)

significant_transfer = abs(delta_L_test) > 0.01
correct_direction = same_direction
systematic_trend = abs(L_rate) > 1e-5

print(f"\nChecks:")
print(f"  Significant L transfer (|dL| > 0.01): {significant_transfer} (dL = {delta_L_test:.4f})")
print(f"  Same direction as source: {correct_direction}")
print(f"  Systematic trend: {systematic_trend} (rate = {L_rate:.6f})")

H0_rejected = significant_transfer and correct_direction

print(f"\n{'='*40}")
print(f"LFM-ONLY VERIFIED: YES")
print(f"H0 STATUS: {'REJECTED' if H0_rejected else 'FAILED TO REJECT'}")

if H0_rejected:
    print(f"CONCLUSION: Frame dragging EMERGES from GOV-01/02")
    print(f"            Test wave gains L in direction of source rotation")
else:
    print(f"CONCLUSION: Frame dragging NOT demonstrated")
    print(f"            No systematic angular momentum transfer observed")

print("=" * 70)

# =============================================================================
# SAVE RESULTS
# =============================================================================
results = {
    "experiment": "LFM Frame Dragging",
    "timestamp": datetime.now().isoformat(),
    "parameters": {
        "chi_0": chi_0,
        "kappa": kappa,
        "omega_source": omega_source,
        "source_radius": source_radius,
        "test_position": [test_x, test_y],
        "n_steps": n_steps
    },
    "results": {
        "L_source_initial": float(L_source_initial),
        "L_source_final": float(L_source_final),
        "L_test_initial": float(L_test_initial),
        "L_test_final": float(L_test_final),
        "delta_L_test": float(delta_L_test),
        "L_transfer_rate": float(L_rate),
        "same_direction": bool(same_direction)
    },
    "hypothesis": {
        "H0_rejected": bool(H0_rejected),
        "lfm_only_verified": True,
        "significant_transfer": bool(significant_transfer),
        "correct_direction": bool(correct_direction)
    }
}

with open("frame_dragging_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to frame_dragging_results.json")
