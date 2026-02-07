#!/usr/bin/env python3
"""
DYNAMIC TAU v2 - Fixed numerical issues
"""

import numpy as np

print("=" * 70)
print("DYNAMIC TAU v2 - Debugging the chi calculation")
print("=" * 70)

# Constants
c = 3e8
H0 = 70e3 / 3.086e22
a_0 = c * H0 / (2 * np.pi)

# Galaxy (NGC6503)
r_kpc = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0])
v_obs = np.array([40, 65, 95, 108, 115, 118, 120, 119, 117])
v_bar = np.array([38, 55, 72, 78, 75, 68, 58, 52, 45])

# Work in kpc and km/s units throughout
chi_0 = 19.0

def compute_dynamic_tau_v2(r_data_kpc, v_bar_kms, chi_0, g_coupling, tau_0_kpc):
    """
    Dynamic tau with proper chi calculation.
    
    Key fix: normalize E^2 properly and track chi changes.
    """
    n = 500
    r_grid = np.linspace(0.1, r_data_kpc[-1] * 2, n)
    dr = r_grid[1] - r_grid[0]
    
    # Interpolate baryonic velocity
    v_bar_grid = np.interp(r_grid, r_data_kpc, v_bar_kms, left=v_bar_kms[0], right=0)
    
    # Baryonic acceleration (in kpc, km/s units)
    # g = v^2/r in (km/s)^2/kpc = 3.086e16 m/s^2 per unit
    # Convert to m/s^2 for comparison with a_0
    kpc_to_m = 3.086e19
    kms_to_ms = 1000
    g_bar_grid = (v_bar_grid * kms_to_ms)**2 / (r_grid * kpc_to_m)  # m/s^2
    g_bar_grid = np.maximum(g_bar_grid, 1e-14)
    
    # Dynamic tau: larger at lower g
    x = g_bar_grid / a_0
    tau_grid_kpc = tau_0_kpc / np.sqrt(np.maximum(x, 0.01))
    
    # Energy density proxy: E^2 ~ v_bar^2
    E_squared = v_bar_grid**2  # (km/s)^2
    E_max = np.max(E_squared)
    E_squared_norm = E_squared / E_max if E_max > 0 else E_squared
    
    # Compute <E^2>_tau with position-dependent tau
    E_squared_avg = np.zeros_like(E_squared_norm)
    for i in range(n):
        tau = tau_grid_kpc[i]
        # Exponential kernel in kpc
        weights = np.exp(-np.abs(r_grid - r_grid[i]) / tau)
        weights = weights / np.sum(weights)
        E_squared_avg[i] = np.sum(weights * E_squared_norm)
    
    # GOV-03: chi^2 = chi_0^2 - g * <E^2>_tau
    # Scale g_coupling so chi varies meaningfully
    chi_squared = chi_0**2 - g_coupling * E_squared_avg * chi_0**2
    chi_squared = np.maximum(chi_squared, 1.0)
    chi = np.sqrt(chi_squared)
    
    # Debug: check chi variation
    print(f"Chi range: {chi.min():.3f} to {chi.max():.3f}")
    print(f"E^2_avg range: {E_squared_avg.min():.4f} to {E_squared_avg.max():.4f}")
    
    # Effective potential: phi = -c^2 * ln(chi/chi_0)
    # Acceleration: a = -d(phi)/dr = c^2 * (dchi/dr) / chi
    dchi_dr = np.gradient(chi, r_grid)
    
    # In natural units with c=1, acceleration = dchi/dr / chi
    # But we need to convert back to km/s^2/kpc
    # Actually, let's compute v^2 = r * a directly
    # If phi ~ -c^2 * delta_chi/chi_0, then
    # a = c^2 * |dchi/dr| / chi
    # v^2 = r * a
    
    # The issue: we need consistent units
    # Let's work in a dimensionless way:
    # Define: v^2 / v_scale^2 = r * (d ln chi / dr)
    
    acceleration_dimless = np.abs(dchi_dr) / chi  # 1/kpc
    v_squared_dimless = r_grid * acceleration_dimless  # dimensionless
    v_dimless = np.sqrt(np.maximum(v_squared_dimless, 0))
    
    return r_grid, chi, v_dimless, tau_grid_kpc, E_squared_avg

# Scan g_coupling
print("\nScanning g_coupling to find good chi variation:")
for g in [0.1, 0.3, 0.5, 0.7, 0.9]:
    print(f"\ng_coupling = {g}:")
    r_grid, chi, v_dim, tau, E2avg = compute_dynamic_tau_v2(r_kpc, v_bar, chi_0, g, 5.0)
    
    # Scale velocity to match observations
    v_at_data = np.interp(r_kpc, r_grid, v_dim)
    if np.max(v_at_data) > 0:
        scale = np.mean(v_obs) / np.mean(v_at_data)
        v_scaled = v_at_data * scale
    else:
        v_scaled = v_at_data
    
    # Metrics
    residuals = v_obs - v_scaled
    rms_pct = np.sqrt(np.mean(residuals**2)) / np.mean(v_obs) * 100
    flatness = np.mean(v_scaled[-3:]) / np.max(v_scaled) if np.max(v_scaled) > 0 else 0
    
    print(f"  RMS: {rms_pct:.1f}%, Flatness: {flatness:.2f}")
    
    if rms_pct < 30:
        print("  r(kpc)  v_obs  v_bar  v_LFM  residual")
        for i in range(len(r_kpc)):
            print(f"  {r_kpc[i]:5.1f}   {v_obs[i]:5.0f}  {v_bar[i]:5.0f}  {v_scaled[i]:5.0f}  {v_obs[i]-v_scaled[i]:+5.0f}")
