#!/usr/bin/env python3
"""
EXPERIMENT: Galaxy Rotation Curves - GOV-03 with Chi Memory
============================================================

GOV-03: chi^2 = chi0^2 - g * <E^2>_tau

The key is the tau-averaging: chi responds to the TIME-AVERAGED
energy density, not instantaneous. This creates:
1. Chi "remembers" where mass WAS
2. Creates extended chi-depression beyond baryonic matter
3. This IS the "dark matter halo" in LFM

For a STATIC galaxy, the tau-average means chi responds to
SPATIALLY-SMOOTHED mass distribution (ergodic-like behavior).
"""

import numpy as np
import json

print("=" * 70)
print("LFM ROTATION CURVES - GOV-03 WITH MEMORY")
print("=" * 70)

# LFM Parameters
chi_0 = 19.0
g = 2.0            # Coupling constant
tau_scale = 15.0   # Memory scale in kpc (spatial equivalent of temporal tau)

print(f"\nParameters: chi_0={chi_0}, g={g}, tau_scale={tau_scale} kpc")

# Test galaxy
GALAXY = {
    "name": "NGC6503",
    "r_data": np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0]),
    "v_obs": np.array([40, 65, 95, 108, 115, 118, 120, 119, 117]),
    "v_bar": np.array([38, 55, 72, 78, 75, 68, 58, 52, 45]),
}

def compute_gov03_rotation(r_data, v_bar, chi_0, g, tau_scale):
    """
    GOV-03 with memory creates extended chi profile.
    
    Key physics:
    1. E^2 ~ v_bar^2 (energy density from baryonic matter)
    2. <E^2>_tau = convolution with exponential kernel
       This EXTENDS the E^2 influence beyond baryonic matter
    3. chi^2 = chi0^2 - g * <E^2>_tau
    4. Acceleration: a = -d(effective_potential)/dr
       where effective potential ~ -c^2 * ln(chi/chi0)
    5. v^2 = r * |a|
    """
    
    # Fine grid
    n = 1000
    r_max = r_data[-1] * 3
    r_grid = np.linspace(0.1, r_max, n)
    dr = r_grid[1] - r_grid[0]
    
    # Interpolate baryonic E^2 (~ v_bar^2)
    v_bar_grid = np.interp(r_grid, r_data, v_bar, left=v_bar[0], right=0)
    E_squared = (v_bar_grid / 100)**2  # Normalize
    
    # TAU-AVERAGING: Convolve with exponential kernel
    # This is the key - chi responds to SMOOTHED E^2
    # For galaxy, interpret tau as spatial scale
    E_squared_avg = np.zeros_like(E_squared)
    for i in range(n):
        # Exponential kernel centered at r_grid[i]
        weights = np.exp(-np.abs(r_grid - r_grid[i]) / tau_scale)
        weights = weights / np.sum(weights)
        E_squared_avg[i] = np.sum(weights * E_squared)
    
    # GOV-03: chi^2 = chi0^2 - g * <E^2>_tau
    chi_squared = chi_0**2 - g * E_squared_avg
    chi_squared = np.maximum(chi_squared, 0.1)  # Prevent negative
    chi = np.sqrt(chi_squared)
    
    # Effective potential: phi = -c^2 * ln(chi/chi0)
    # Acceleration: a = -d(phi)/dr = c^2 * (1/chi) * dchi/dr
    dchi_dr = np.gradient(chi, r_grid)
    
    # a = c^2 * (dchi/dr) / chi  (with c=1)
    # For attraction (toward center where chi is lower):
    # chi decreases toward center, so dchi/dr > 0 at large r
    # This means a > 0 (pointing inward for our convention)
    
    acceleration = np.abs(dchi_dr) / chi
    
    # Circular velocity
    v_squared = r_grid * acceleration
    v_lfm = np.sqrt(np.maximum(v_squared, 0))
    
    return r_grid, chi, v_lfm, E_squared, E_squared_avg

# Compute
r_grid, chi, v_lfm, E2, E2_avg = compute_gov03_rotation(
    GALAXY["r_data"], GALAXY["v_bar"], chi_0, g, tau_scale
)

# Get at data points and scale
v_lfm_data = np.interp(GALAXY["r_data"], r_grid, v_lfm)
scale = np.max(GALAXY["v_obs"]) / np.max(v_lfm_data) if np.max(v_lfm_data) > 0 else 1
v_lfm_scaled = v_lfm_data * scale

# Results
print(f"\n{GALAXY['name']}:")
print(f"  r(kpc)  v_obs  v_bar  v_LFM  resid")
for i in range(len(GALAXY["r_data"])):
    r = GALAXY["r_data"][i]
    vo = GALAXY["v_obs"][i]
    vb = GALAXY["v_bar"][i]
    vl = v_lfm_scaled[i]
    print(f"  {r:5.1f}   {vo:5.0f}  {vb:5.0f}  {vl:5.0f}  {vo-vl:+5.0f}")

# Flatness check
v_outer = np.mean(v_lfm_scaled[-3:])
v_peak = np.max(v_lfm_scaled)
flatness = v_outer / v_peak
print(f"\nFlatness: {flatness:.2f} ({'FLAT' if flatness > 0.85 else 'DECLINING'})")

# RMS error
residuals = GALAXY["v_obs"] - v_lfm_scaled
rms = np.sqrt(np.mean(residuals**2))
rms_pct = rms / np.mean(GALAXY["v_obs"]) * 100
print(f"RMS error: {rms_pct:.1f}%")

# Chi profile
print(f"\nChi profile:")
for r in [1, 5, 10, 15, 20, 30]:
    idx = np.argmin(np.abs(r_grid - r))
    print(f"  r={r:2d} kpc: chi={chi[idx]:.3f}, E2={E2[idx]:.4f}, E2_avg={E2_avg[idx]:.4f}")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

if flatness > 0.85:
    print("""
SUCCESS: GOV-03 with tau-memory produces flat rotation curve!

THE MECHANISM:
1. Tau-averaging smooths E^2 over spatial scale tau_scale
2. This EXTENDS chi depression beyond baryonic matter
3. Extended chi profile creates "dark matter halo" effect
4. Chi gradient at large r doesn't fall off as fast as baryonic mass

This is the LFM explanation for dark matter: it's chi MEMORY,
not particles.
""")
else:
    print(f"""
GOV-03 with tau_scale={tau_scale} kpc still produces declining curve.

POSSIBILITIES:
1. tau_scale needs tuning (try larger values)
2. Need full GOV-02 wave dynamics, not just GOV-03 approximation
3. Need cosmological chi_0 evolution (a_0 = cH_0/2pi coupling)

Let me try different tau_scale values...
""")
    
    print("\n" + "=" * 70)
    print("PARAMETER SCAN: tau_scale")
    print("=" * 70)
    
    for tau in [5, 10, 20, 30, 50, 100]:
        r_grid, chi, v_lfm, _, _ = compute_gov03_rotation(
            GALAXY["r_data"], GALAXY["v_bar"], chi_0, g, tau
        )
        v_lfm_data = np.interp(GALAXY["r_data"], r_grid, v_lfm)
        scale = np.max(GALAXY["v_obs"]) / np.max(v_lfm_data) if np.max(v_lfm_data) > 0 else 1
        v_lfm_scaled = v_lfm_data * scale
        
        v_outer = np.mean(v_lfm_scaled[-3:])
        v_peak = np.max(v_lfm_scaled)
        flatness = v_outer / v_peak
        
        residuals = GALAXY["v_obs"] - v_lfm_scaled
        rms_pct = np.sqrt(np.mean(residuals**2)) / np.mean(GALAXY["v_obs"]) * 100
        
        status = "FLAT" if flatness > 0.85 else "decline"
        print(f"  tau={tau:3d} kpc: flatness={flatness:.2f} ({status}), RMS={rms_pct:.1f}%")

print("\n" + "=" * 70)
