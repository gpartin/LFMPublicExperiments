#!/usr/bin/env python3
"""
EXPERIMENT: Galaxy Rotation Curves - PROPER GOV-04 Derivation
==============================================================

THIS TIME: Actually solve GOV-04 for chi profile, then derive velocity.

PHYSICS:
1. GOV-04 (quasi-static): nabla^2 chi = (kappa/c^2) * (E^2 - E0^2)
2. For spherical symmetry: (1/r^2) d/dr (r^2 dchi/dr) = kappa * rho(r)
3. Acceleration: a = -c^2 * d(ln chi)/dr = -c^2 * (dchi/dr) / chi
4. Circular velocity: v^2 = r * |a|

NO borrowed RAR formula. Let's see what GOV-04 actually predicts.
"""

import numpy as np
import json
from datetime import datetime

print("=" * 70)
print("LFM ROTATION CURVES - PROPER GOV-04 SOLUTION")
print("NO empirical RAR - pure chi dynamics")
print("=" * 70)

# =============================================================================
# LFM PARAMETERS
# =============================================================================
chi_0 = 19.0       # Background chi (fundamental)
kappa = 0.016      # Chi-E^2 coupling
c = 1.0            # Natural units

print(f"\nLFM Parameters: chi_0 = {chi_0}, kappa = {kappa}")

# =============================================================================
# SPARC DATA (embedded sample)
# =============================================================================
GALAXIES = {
    "NGC6503": {"type": "Sc spiral", "data": np.array([
        [0.5, 40, 5, 38], [1.0, 65, 4, 55], [2.0, 95, 4, 72],
        [3.0, 108, 3, 78], [5.0, 115, 3, 75], [7.0, 118, 4, 68],
        [10.0, 120, 5, 58], [12.0, 119, 6, 52], [15.0, 117, 8, 45],
    ])},
    "UGC128": {"type": "LSB dwarf", "data": np.array([
        [2.0, 55, 8, 25], [5.0, 85, 6, 40], [10.0, 105, 5, 50],
        [15.0, 115, 5, 52], [20.0, 120, 6, 50], [25.0, 122, 7, 47],
        [30.0, 120, 8, 43],
    ])},
    "DDO154": {"type": "Dwarf irregular", "data": np.array([
        [0.5, 15, 3, 8], [1.0, 28, 3, 15], [2.0, 40, 3, 22],
        [3.0, 47, 3, 25], [4.0, 50, 4, 26], [5.0, 52, 5, 25],
        [6.0, 53, 6, 24],
    ])},
}

# =============================================================================
# SOLVE GOV-04 FOR CHI PROFILE
# =============================================================================

def solve_gov04_chi_profile(r_data, v_bar, chi_0, kappa):
    """
    Solve GOV-04 for chi(r) given baryonic mass distribution.
    
    GOV-04: nabla^2 chi = (kappa/c^2) * E^2
    
    For spherical symmetry:
    (1/r^2) d/dr (r^2 dchi/dr) = kappa * rho(r)
    
    Integrating once:
    r^2 dchi/dr = kappa * integral_0^r rho(r') r'^2 dr' = kappa * M(<r) / (4*pi)
    
    dchi/dr = kappa * M(<r) / (4*pi * r^2)
    
    Integrating again:
    chi(r) = chi_0 - kappa * integral_r^inf M(<r') / (4*pi * r'^2) dr'
    
    For finite galaxy, use:
    chi(r) = chi_0 - kappa * sum over shells
    """
    
    # Create fine radial grid
    n = 500
    r_max = r_data[-1] * 3
    r_grid = np.linspace(0.1, r_max, n)
    dr = r_grid[1] - r_grid[0]
    
    # Interpolate v_bar to get baryonic velocity profile
    v_bar_grid = np.interp(r_grid, r_data, v_bar, left=v_bar[0], right=0)
    
    # Infer enclosed mass from v_bar: M(<r) ~ v_bar^2 * r
    # (virial theorem: v^2 = GM/r => M = v^2 * r)
    M_enclosed = v_bar_grid**2 * r_grid
    
    # Normalize to reasonable values
    M_max = np.max(M_enclosed)
    if M_max > 0:
        M_enclosed = M_enclosed / M_max
    
    # Solve for chi gradient: dchi/dr = kappa * M(<r) / (4*pi * r^2)
    # This is the Green's function solution of Poisson equation
    dchi_dr = np.zeros_like(r_grid)
    for i, r in enumerate(r_grid):
        if r > 0:
            dchi_dr[i] = kappa * M_enclosed[i] / (4 * np.pi * r**2)
    
    # Chi gradient is OUTWARD (chi decreases toward mass)
    # So dchi/dr should be negative (chi lower near mass)
    dchi_dr = -dchi_dr
    
    # Integrate to get chi(r) from boundary condition chi(inf) = chi_0
    # chi(r) = chi_0 + integral_r^inf (dchi/dr') dr'
    chi = np.zeros_like(r_grid)
    chi[-1] = chi_0  # Boundary at infinity
    for i in range(n-2, -1, -1):
        chi[i] = chi[i+1] - dchi_dr[i+1] * dr
    
    # Ensure chi stays positive
    chi = np.maximum(chi, 0.1)
    
    return r_grid, chi, M_enclosed

def compute_rotation_from_chi(r_grid, chi):
    """
    Compute circular velocity from chi gradient.
    
    Acceleration: a = -c^2 * d(ln chi)/dr = -c^2 * (dchi/dr) / chi
    
    For circular orbit: v^2 = r * |a|
    """
    
    # Compute chi gradient
    dchi_dr = np.gradient(chi, r_grid)
    
    # Acceleration from chi gradient
    # a = -c^2 * (dchi/dr) / chi
    # With c=1 in natural units:
    a = -(dchi_dr) / chi
    
    # Circular velocity: v^2 = r * |a|
    v_squared = r_grid * np.abs(a)
    v = np.sqrt(np.maximum(v_squared, 0))
    
    return v, a

# =============================================================================
# FIT GALAXIES
# =============================================================================
print("\n" + "=" * 70)
print("SOLVING GOV-04 FOR EACH GALAXY")
print("=" * 70)

results = []
for name, galaxy in GALAXIES.items():
    data = galaxy["data"]
    r_data = data[:, 0]
    v_obs = data[:, 1]
    v_err = data[:, 2]
    v_bar = data[:, 3]
    
    # Solve GOV-04
    r_grid, chi_profile, M_enc = solve_gov04_chi_profile(r_data, v_bar, chi_0, kappa)
    
    # Get velocity from chi
    v_lfm_raw, a_profile = compute_rotation_from_chi(r_grid, chi_profile)
    
    # Interpolate to data points
    v_lfm_data = np.interp(r_data, r_grid, v_lfm_raw)
    
    # Scale to match observations (unit conversion)
    if np.max(v_lfm_data) > 0:
        scale = np.max(v_obs) / np.max(v_lfm_data)
    else:
        scale = 1.0
    v_lfm_scaled = v_lfm_data * scale
    
    # Metrics
    residuals = v_obs - v_lfm_scaled
    rms = np.sqrt(np.mean(residuals**2))
    rms_pct = rms / np.mean(v_obs) * 100
    
    # Flatness check
    v_outer = np.mean(v_lfm_scaled[-3:]) if len(v_lfm_scaled) >= 3 else v_lfm_scaled[-1]
    v_peak = np.max(v_lfm_scaled)
    flatness = v_outer / v_peak if v_peak > 0 else 0
    is_flat = flatness > 0.85
    
    result = {
        "name": name,
        "rms_pct": float(rms_pct),
        "flatness": float(flatness),
        "is_flat": bool(is_flat),
        "r_data": r_data.tolist(),
        "v_obs": v_obs.tolist(),
        "v_bar": v_bar.tolist(),
        "v_lfm": v_lfm_scaled.tolist(),
    }
    results.append(result)
    
    print(f"\n{name} ({galaxy['type']}):")
    print(f"  RMS error: {rms_pct:.1f}%")
    print(f"  Flatness: {flatness:.2f} ({'FLAT' if is_flat else 'DECLINING'})")
    print(f"  Chi at center: {chi_profile[0]:.2f}, at edge: {chi_profile[-1]:.2f}")
    print(f"  r(kpc)  v_obs  v_bar  v_LFM  residual")
    for i in range(len(r_data)):
        print(f"  {r_data[i]:5.1f}   {v_obs[i]:5.0f}  {v_bar[i]:5.0f}  {v_lfm_scaled[i]:5.0f}  {v_obs[i]-v_lfm_scaled[i]:+6.0f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY - PURE GOV-04 (NO RAR FORMULA)")
print("=" * 70)

n_flat = sum(1 for r in results if r['is_flat'])
avg_error = np.mean([r['rms_pct'] for r in results])

print(f"\nAcross {len(results)} galaxies:")
print(f"  Flat curves: {n_flat}/{len(results)}")
print(f"  Average RMS error: {avg_error:.1f}%")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if n_flat < len(results) // 2:
    print("""
RESULT: GOV-04 alone does NOT produce flat rotation curves!

PHYSICS EXPLANATION:
- GOV-04: nabla^2 chi = kappa * E^2
- This is structurally identical to Poisson's equation
- Solution: chi(r) ~ chi_0 - M(<r)/r (like Newtonian potential)
- Velocity: v ~ sqrt(M(<r)/r) -> Keplerian decline at large r

THE PROBLEM:
- GOV-04 is the QUASI-STATIC limit
- It gives Newtonian gravity, which has Keplerian decline
- Flat curves require ADDITIONAL physics

WHAT'S MISSING:
1. Chi MEMORY (tau-averaging in GOV-03) - creates extended halos
2. Chi WAVE dynamics (full GOV-02) - non-static effects
3. Cosmological coupling (chi_0 evolution) - creates a0 scale

HONEST CONCLUSION:
- Pure GOV-04 -> Newtonian gravity -> Keplerian decline
- The RAR relation is NOT a trivial consequence of GOV-04
- LFM needs GOV-03 (memory) or full GOV-01+02 dynamics for flat curves
""")
else:
    print("GOV-04 produces flat curves - unexpected!")

print("=" * 70)
