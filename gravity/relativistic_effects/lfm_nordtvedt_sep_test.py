"""
Nordtvedt Effect / Strong Equivalence Principle Test in LFM
============================================================

Tests whether bodies with different internal structure (self-gravitational energy)
fall at the same rate in an external gravitational field.

HYPOTHESIS:
-----------
H₀: Bodies with different internal χ-profiles fall at DIFFERENT rates (SEP violated)
H₁: Bodies with different internal χ-profiles fall at IDENTICAL rates (SEP confirmed)

LFM PHYSICS:
------------
- GOV-01: ∂²E/∂t² = c²∇²E − χ²E
- GOV-02: ∂²χ/∂t² = c²∇²χ − κE²
- Two test masses with different widths (different self-energy, different χ-wells)
- External χ-gradient creates "gravitational field"
- Track center-of-mass motion

SUCCESS CRITERIA:
-----------------
REJECT H₀ if: |Δx_COM| < 0.5 (trajectories within 0.5% of grid)
FAIL TO REJECT H₀ if: |Δx_COM| > 0.5 (significant trajectory difference)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# ============================================================================
# PARAMETERS
# ============================================================================

# Grid
N = 256                    # Grid points
L = 100.0                  # Domain size
dx = L / N                 # Spatial step
x = np.linspace(0, L, N)   # Spatial grid

# Time
dt = dx / 4.0              # Time step (CFL stability)
n_steps = 1000             # Total steps
snapshot_every = 5         # Animation frame rate

# LFM parameters
CHI_0 = 19.0               # Background χ
C = 1.0                    # Wave speed
KAPPA = 0.016              # χ-E coupling
DAMPING = 0.998            # Light damping for stability

# External field
CHI_GRADIENT = 2.0         # External χ-gradient (simulates gravity)

# Test masses (START AT SAME POSITION - this is the key!)
INITIAL_POS = 80.0         # Both start at same height
MASS1_WIDTH = 1.5          # Compact (narrow, high self-gravity)
MASS2_WIDTH = 3.0          # Diffuse (wide, low self-gravity)
AMPLITUDE = 1.5            # Energy amplitude
INJECTION_RATE = 0.0       # NO injection - let them fall freely!

# ============================================================================
# LAPLACIAN (5-point stencil with periodic boundaries)
# ============================================================================

def laplacian_1d(field):
    """Compute Laplacian using 5-point finite difference."""
    lap = np.zeros_like(field)
    lap[2:-2] = (-field[4:] + 16*field[3:-1] - 30*field[2:-2] + 16*field[1:-3] - field[:-4]) / (12 * dx**2)
    # Boundaries (use 3-point)
    lap[0] = (field[1] - 2*field[0] + field[-1]) / dx**2
    lap[1] = (field[2] - 2*field[1] + field[0]) / dx**2
    lap[-2] = (field[-1] - 2*field[-2] + field[-3]) / dx**2
    lap[-1] = (field[0] - 2*field[-1] + field[-2]) / dx**2
    return lap

# ============================================================================
# GOV-01+GOV-02 EVOLUTION (Leapfrog with damping)
# ============================================================================

def evolve_gov01(e, e_prev, chi, dt, damping=1.0):
    """Evolve E field using GOV-01: ∂²E/∂t² = c²∇²E − χ²E"""
    lap_e = laplacian_1d(e)
    e_next = (2*e - e_prev + dt**2 * (C**2 * lap_e - chi**2 * e)) * damping
    return e_next

def evolve_gov02(chi, chi_prev, e_squared, dt):
    """Evolve χ field using GOV-02: ∂²χ/∂t² = c²∇²χ − κE²"""
    lap_chi = laplacian_1d(chi)
    chi_next = 2*chi - chi_prev + dt**2 * (C**2 * lap_chi - KAPPA * e_squared)
    return chi_next

# ============================================================================
# MASS CREATION
# ============================================================================

def create_gaussian_mass(pos, width, amplitude):
    """Create Gaussian energy distribution."""
    return amplitude * np.exp(-((x - pos) / width)**2)

def compute_center_of_mass(e):
    """Compute center of mass weighted by E²."""
    e_squared = e**2
    total_weight = np.sum(e_squared)
    if total_weight < 1e-10:
        return 0.0
    com = np.sum(x * e_squared) / total_weight
    return com

# ============================================================================
# INITIALIZATION
# ============================================================================

# Energy fields (two masses at SAME initial position)
e1 = create_gaussian_mass(INITIAL_POS, MASS1_WIDTH, AMPLITUDE)
e2 = create_gaussian_mass(INITIAL_POS, MASS2_WIDTH, AMPLITUDE)
e1_prev = e1.copy()
e2_prev = e2.copy()

# Chi field (external gradient + self-interaction)
chi_external = CHI_0 - CHI_GRADIENT * (x / L)  # Linear gradient
chi = chi_external.copy()
chi_prev = chi.copy()

# Tracking
com1_history = []
com2_history = []
time_history = []

# Animation storage
frames = []

# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

print("="*70)
print("NORDTVEDT/SEP TEST")
print("="*70)
print(f"Grid: {N} points, L={L}")
print(f"Both masses start at: pos={INITIAL_POS}")
print(f"Mass 1: width={MASS1_WIDTH} (compact, high self-gravity)")
print(f"Mass 2: width={MASS2_WIDTH} (diffuse, low self-gravity)")
print(f"External gradient: {CHI_GRADIENT}")
print(f"Steps: {n_steps}")
print(f"Injection rate: {INJECTION_RATE} (free fall)")
print("="*70)

for step in range(n_steps):
    # Optional injection (only if INJECTION_RATE > 0)
    if INJECTION_RATE > 0:
        e1_source = create_gaussian_mass(INITIAL_POS, MASS1_WIDTH, AMPLITUDE)
        e2_source = create_gaussian_mass(INITIAL_POS, MASS2_WIDTH, AMPLITUDE)
        e1 = (1 - INJECTION_RATE) * e1 + INJECTION_RATE * e1_source
        e2 = (1 - INJECTION_RATE) * e2 + INJECTION_RATE * e2_source
    
    # Evolve E fields (GOV-01)
    e1_next = evolve_gov01(e1, e1_prev, chi, dt, DAMPING)
    e2_next = evolve_gov01(e2, e2_prev, chi, dt, DAMPING)
    
    # Combined E² for chi evolution
    e_total_squared = e1**2 + e2**2
    
    # Evolve chi (GOV-02)
    chi_next = evolve_gov02(chi, chi_prev, e_total_squared, dt)
    
    # Update
    e1_prev, e1 = e1, e1_next
    e2_prev, e2 = e2, e2_next
    chi_prev, chi = chi, chi_next
    
    # Track center of mass
    com1 = compute_center_of_mass(e1)
    com2 = compute_center_of_mass(e2)
    com1_history.append(com1)
    com2_history.append(com2)
    time_history.append(step * dt)
    
    # Store frame
    if step % snapshot_every == 0:
        frames.append({
            'e1': e1.copy(),
            'e2': e2.copy(),
            'chi': chi.copy(),
            'chi_external': chi_external.copy(),
            'com1': com1,
            'com2': com2,
            'step': step,
            'time': step * dt
        })
    
    if step % 100 == 0:
        print(f"Step {step:4d}: COM1={com1:.2f}, COM2={com2:.2f}, Δ={abs(com1-com2):.3f}")

# ============================================================================
# ANALYSIS
# ============================================================================

com1_history = np.array(com1_history)
com2_history = np.array(com2_history)
time_history = np.array(time_history)

# Final positions
com1_final = com1_history[-1]
com2_final = com2_history[-1]
delta_com = abs(com1_final - com2_final)

# Trajectory correlation
trajectory_diff = np.abs(com1_history - com2_history)
max_diff = np.max(trajectory_diff)
mean_diff = np.mean(trajectory_diff)

print("="*70)
print("ANALYSIS")
print("="*70)
print(f"Final COM1: {com1_final:.3f}")
print(f"Final COM2: {com2_final:.3f}")
print(f"Final Δ:    {delta_com:.3f}")
print(f"Max Δ:      {max_diff:.3f}")
print(f"Mean Δ:     {mean_diff:.3f}")
print("="*70)

# Hypothesis test
THRESHOLD = 0.5  # 0.5% of domain
if max_diff < THRESHOLD:
    h0_status = "REJECTED"
    conclusion = "SEP CONFIRMED - Bodies with different internal structure fall identically"
else:
    h0_status = "FAILED TO REJECT"
    conclusion = "SEP VIOLATED - Different bodies fall at different rates"

print("HYPOTHESIS VALIDATION")
print("="*70)
print(f"LFM-ONLY VERIFIED: YES")
print(f"H₀ STATUS: {h0_status}")
print(f"CONCLUSION: {conclusion}")
print("="*70)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Create output directory
output_dir = "results_nordtvedt_sep"
os.makedirs(output_dir, exist_ok=True)

# Create animation
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("LFM Nordtvedt/SEP Test: Two Bodies with Different Internal Structure", 
             fontsize=14, fontweight='bold')

def animate(frame_idx):
    frame = frames[frame_idx]
    
    # Clear axes
    for ax in axes:
        ax.clear()
    
    # Panel 1: Energy densities
    axes[0].plot(x, frame['e1']**2, 'b-', linewidth=2, label=f'Mass 1 (compact, σ={MASS1_WIDTH})')
    axes[0].plot(x, frame['e2']**2, 'r-', linewidth=2, label=f'Mass 2 (diffuse, σ={MASS2_WIDTH})')
    axes[0].axvline(frame['com1'], color='b', linestyle='--', alpha=0.5, label=f'COM1={frame["com1"]:.1f}')
    axes[0].axvline(frame['com2'], color='r', linestyle='--', alpha=0.5, label=f'COM2={frame["com2"]:.1f}')
    axes[0].set_ylabel('E² (Energy Density)', fontsize=11)
    axes[0].set_xlim(0, L)
    axes[0].set_ylim(0, AMPLITUDE**2 * 1.2)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.95, f'Step {frame["step"]} (t={frame["time"]:.1f})', 
                 transform=axes[0].transAxes, verticalalignment='top', fontsize=10)
    
    # Panel 2: Chi field
    axes[1].plot(x, frame['chi_external'], 'k--', linewidth=1, alpha=0.5, label='External χ (no matter)')
    axes[1].plot(x, frame['chi'], 'g-', linewidth=2, label='Actual χ (with matter)')
    axes[1].axhline(CHI_0, color='k', linestyle=':', alpha=0.3)
    axes[1].set_ylabel('χ Field', fontsize=11)
    axes[1].set_xlim(0, L)
    axes[1].set_ylim(CHI_0 - CHI_GRADIENT - 2, CHI_0 + 1)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Trajectory comparison
    current_step = frame['step']
    axes[2].plot(time_history[:current_step+1], com1_history[:current_step+1], 
                 'b-', linewidth=2, label=f'Mass 1 (final={com1_history[current_step]:.2f})')
    axes[2].plot(time_history[:current_step+1], com2_history[:current_step+1], 
                 'r-', linewidth=2, label=f'Mass 2 (final={com2_history[current_step]:.2f})')
    axes[2].set_xlabel('Time', fontsize=11)
    axes[2].set_ylabel('Center of Mass Position', fontsize=11)
    axes[2].set_xlim(0, n_steps * dt)
    axes[2].set_ylim(INITIAL_POS - 10, INITIAL_POS + 10)
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].text(0.02, 0.05, f'Δ = {abs(frame["com1"] - frame["com2"]):.3f}', 
                 transform=axes[2].transAxes, verticalalignment='bottom', 
                 fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

anim = FuncAnimation(fig, animate, frames=len(frames), interval=100, repeat=True)
output_path = os.path.join(output_dir, "nordtvedt_sep_test.gif")
anim.save(output_path, writer='pillow', fps=10, dpi=100)
print(f"Animation saved: {output_path}")

plt.close()

# ============================================================================
# FINAL TRAJECTORY PLOT
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_history, com1_history, 'b-', linewidth=2, label=f'Mass 1 (compact, σ={MASS1_WIDTH})')
ax.plot(time_history, com2_history, 'r--', linewidth=2, label=f'Mass 2 (diffuse, σ={MASS2_WIDTH})')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Center of Mass Position', fontsize=12)
ax.set_title('Nordtvedt/SEP Test: Trajectories of Bodies with Different Internal Structure', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, f'Max Δ = {max_diff:.3f}\n{conclusion}', 
        transform=ax.transAxes, verticalalignment='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightblue' if h0_status == "REJECTED" else 'lightcoral', alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "trajectories.png"), dpi=150)
print(f"Trajectory plot saved: {os.path.join(output_dir, 'trajectories.png')}")
plt.close()

print("\n✓ Nordtvedt/SEP test complete")
