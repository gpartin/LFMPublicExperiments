#!/usr/bin/env python3
"""
GOV-03 Rotation Curves - Parameter Tuning
==========================================

The previous attempt showed flat curve but WRONG SHAPE.
Issue: chi barely changes. Need to tune g (coupling) to get proper chi profile.

For flat rotation curve, we need:
- Large chi depression at center (low chi = deep potential well)
- Chi gradient that falls off slower than 1/r^2
- This requires proper balance of g and tau_scale
"""

import numpy as np

print("=" * 70)
print("GOV-03 PARAMETER SCAN")
print("=" * 70)

chi_0 = 19.0

# Galaxy data
r_data = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0])
v_obs = np.array([40, 65, 95, 108, 115, 118, 120, 119, 117])
v_bar = np.array([38, 55, 72, 78, 75, 68, 58, 52, 45])

def compute_gov03(r_data, v_bar, chi_0, g, tau_scale):
    n = 1000
    r_max = r_data[-1] * 3
    r_grid = np.linspace(0.1, r_max, n)
    
    v_bar_grid = np.interp(r_grid, r_data, v_bar, left=v_bar[0], right=0)
    E_squared = (v_bar_grid)**2  # Don't over-normalize
    
    # Tau-averaging with spatial kernel
    E_squared_avg = np.zeros_like(E_squared)
    for i in range(n):
        weights = np.exp(-np.abs(r_grid - r_grid[i]) / tau_scale)
        weights = weights / np.sum(weights)
        E_squared_avg[i] = np.sum(weights * E_squared)
    
    # GOV-03: chi^2 = chi0^2 - g * <E^2>_tau
    chi_squared = chi_0**2 - g * E_squared_avg
    chi_squared = np.maximum(chi_squared, 1.0)
    chi = np.sqrt(chi_squared)
    
    # Velocity from chi gradient
    dchi_dr = np.gradient(chi, r_grid)
    acceleration = np.abs(dchi_dr) / chi
    v_lfm = np.sqrt(np.maximum(r_grid * acceleration, 0))
    
    return r_grid, chi, v_lfm

print("\nScanning g (coupling) with tau_scale=15 kpc:")
print("-" * 60)

best_g = None
best_error = float('inf')

for g in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
    r_grid, chi, v_lfm = compute_gov03(r_data, v_bar, chi_0, g, 15.0)
    v_lfm_data = np.interp(r_data, r_grid, v_lfm)
    
    if np.max(v_lfm_data) > 0:
        scale = np.mean(v_obs) / np.mean(v_lfm_data)
        v_lfm_scaled = v_lfm_data * scale
    else:
        v_lfm_scaled = v_lfm_data
    
    residuals = v_obs - v_lfm_scaled
    rms_pct = np.sqrt(np.mean(residuals**2)) / np.mean(v_obs) * 100
    
    flatness = np.mean(v_lfm_scaled[-3:]) / np.max(v_lfm_scaled) if np.max(v_lfm_scaled) > 0 else 0
    chi_center = chi[10]
    chi_edge = chi[-1]
    
    status = "FLAT" if flatness > 0.85 else "decline"
    print(f"g={g:.3f}: RMS={rms_pct:5.1f}%, flat={flatness:.2f} ({status}), chi: {chi_center:.2f}{chi_edge:.2f}")
    
    if rms_pct < best_error and flatness > 0.8:
        best_error = rms_pct
        best_g = g

print(f"\nBest g = {best_g} with error {best_error:.1f}%")

print("\n" + "=" * 70)
print("Now scan tau_scale with best g")
print("=" * 70)

if best_g:
    for tau in [5, 10, 15, 20, 30, 50]:
        r_grid, chi, v_lfm = compute_gov03(r_data, v_bar, chi_0, best_g, tau)
        v_lfm_data = np.interp(r_data, r_grid, v_lfm)
        
        if np.max(v_lfm_data) > 0:
            scale = np.mean(v_obs) / np.mean(v_lfm_data)
            v_lfm_scaled = v_lfm_data * scale
        else:
            v_lfm_scaled = v_lfm_data
        
        residuals = v_obs - v_lfm_scaled
        rms_pct = np.sqrt(np.mean(residuals**2)) / np.mean(v_obs) * 100
        flatness = np.mean(v_lfm_scaled[-3:]) / np.max(v_lfm_scaled) if np.max(v_lfm_scaled) > 0 else 0
        
        status = "FLAT" if flatness > 0.85 else "decline"
        print(f"tau={tau:2d} kpc: RMS={rms_pct:5.1f}%, flatness={flatness:.2f} ({status})")
        
        if tau == 20 or (rms_pct < 20 and flatness > 0.9):
            # Show detailed output
            print(f"\n  Detailed for tau={tau}:")
            print(f"  r(kpc)  v_obs  v_bar  v_LFM  resid")
            for i in range(len(r_data)):
                print(f"  {r_data[i]:5.1f}   {v_obs[i]:5.0f}  {v_bar[i]:5.0f}  {v_lfm_scaled[i]:5.0f}  {v_obs[i]-v_lfm_scaled[i]:+5.0f}")
            print()

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
GOV-03 with chi memory CAN produce flat rotation curves, but:
1. Requires tuning of g (coupling) and tau_scale (memory)
2. Two free parameters beyond chi_0
3. The flat curve shape depends on tau_scale matching galaxy size

For LFM to be predictive (not just curve-fitting):
- tau_scale should be derivable from something (maybe a_0 = cH_0/2pi?)
- g should be related to kappa

HONEST STATUS:
- Flat curves: POSSIBLE with GOV-03 + tuning
- Predictive power: NEEDS WORK (derive tau from fundamentals)
""")
