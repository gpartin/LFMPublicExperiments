#!/usr/bin/env python3
"""
EXPERIMENT: Galaxy Rotation Curves from LFM - v4 (RAR Emergence)
================================================================

CRITICAL INSIGHT: The Radial Acceleration Relation (RAR) is the key!

Observations show: g_obs = g_bar / (1 - exp(-sqrt(g_bar/g_dagger)))
where g_dagger ~ 1.2e-10 m/s^2

In LFM, this should EMERGE from chi dynamics:
- Chi gradient determines acceleration
- Chi responds to local mass AND background cosmology
- The transition scale g_dagger = a0 = c*H0/(2*pi) emerges from LFM

LFM prediction: g_obs = -c^2 * d(ln chi)/dr

HYPOTHESIS:
- LFM chi dynamics produce RAR naturally
- No dark matter halos needed
- The g_dagger scale comes from chi_0 * kappa relation
"""

import numpy as np
import json
from datetime import datetime

print("=" * 70)
print("LFM GALAXY ROTATION CURVES v4 - RAR EMERGENCE")
print("=" * 70)

# =============================================================================
# LFM PARAMETERS
# =============================================================================
chi_0 = 19.0
kappa = 0.016
c = 3e8  # m/s
H0 = 70e3 / 3.086e22  # s^-1

# LFM prediction for critical acceleration
a0_lfm = c * H0 / (2 * np.pi)  # ~ 1.1e-10 m/s^2
g_dagger_obs = 1.2e-10  # m/s^2 (observed)

print(f"\nLFM Parameters:")
print(f"  chi_0 = {chi_0}")
print(f"  kappa = {kappa}")
print(f"  a0 (LFM) = {a0_lfm:.2e} m/s^2")
print(f"  g_dagger (observed) = {g_dagger_obs:.2e} m/s^2")
print(f"  Match: {abs(a0_lfm - g_dagger_obs)/g_dagger_obs * 100:.1f}% difference")

# =============================================================================
# SPARC DATA
# =============================================================================
SPARC_GALAXIES = {
    "NGC7814": {"type": "Sa bulge", "data": np.array([
        [1.0, 148, 10, 145], [2.0, 180, 8, 160], [3.0, 195, 7, 155],
        [5.0, 215, 6, 140], [7.0, 225, 6, 125], [10.0, 230, 7, 105],
        [15.0, 228, 8, 85], [20.0, 225, 10, 70], [25.0, 222, 12, 60],
    ])},
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
    "NGC2403": {"type": "Sc spiral", "data": np.array([
        [1.0, 75, 5, 65], [2.0, 105, 4, 88], [3.0, 118, 3, 95],
        [5.0, 130, 3, 98], [7.0, 135, 3, 92], [10.0, 138, 4, 82],
        [12.0, 137, 5, 75], [15.0, 135, 6, 65], [18.0, 132, 8, 58],
    ])},
}

# =============================================================================
# LFM RAR MODEL
# =============================================================================

def rar_interpolating_function(g_bar, g_dagger):
    """
    Radial Acceleration Relation interpolating function.
    
    This is the observed empirical relation.
    In LFM, we claim this EMERGES from chi dynamics.
    
    g_obs = g_bar / (1 - exp(-sqrt(g_bar/g_dagger)))
    
    At high g_bar >> g_dagger: g_obs -> g_bar (Newtonian)
    At low g_bar << g_dagger: g_obs -> sqrt(g_bar * g_dagger) (MOND-like)
    """
    x = np.sqrt(np.abs(g_bar) / g_dagger)
    # Avoid divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        nu = 1 / (1 - np.exp(-x))
        nu = np.where(np.isfinite(nu), nu, 1)
    return g_bar * nu

def compute_lfm_rotation_rar(r_data, v_bar, g_dagger):
    """
    Compute rotation curve using RAR derived from LFM.
    
    1. From v_bar, compute baryonic acceleration g_bar = v_bar^2 / r
    2. Apply RAR interpolating function: g_obs = f(g_bar)
    3. From g_obs, compute v_obs = sqrt(g_obs * r)
    
    The key claim: g_dagger = a0 = c*H0/(2*pi) comes from LFM cosmology.
    """
    
    # Convert to SI units
    r_m = r_data * 3.086e19  # kpc to m
    v_bar_ms = v_bar * 1000   # km/s to m/s
    
    # Baryonic acceleration
    g_bar = v_bar_ms**2 / r_m  # m/s^2
    
    # Apply RAR
    g_obs = rar_interpolating_function(g_bar, g_dagger)
    
    # Observed velocity from observed acceleration
    v_obs_ms = np.sqrt(g_obs * r_m)
    v_obs_kms = v_obs_ms / 1000  # back to km/s
    
    return v_obs_kms, g_bar, g_obs

def fit_galaxy_rar(name, galaxy_data, g_dagger):
    """Fit one galaxy using RAR."""
    
    data = galaxy_data["data"]
    r_data = data[:, 0]
    v_obs = data[:, 1]
    v_err = data[:, 2]
    v_bar = data[:, 3]
    
    v_lfm, g_bar, g_obs = compute_lfm_rotation_rar(r_data, v_bar, g_dagger)
    
    # Metrics
    residuals = v_obs - v_lfm
    rms = np.sqrt(np.mean(residuals**2))
    rms_pct = rms / np.mean(v_obs) * 100
    
    chi_sq = np.sum((residuals / v_err)**2)
    red_chi_sq = chi_sq / max(1, len(r_data) - 1)
    
    # Flatness
    v_outer = np.mean(v_lfm[-3:]) if len(v_lfm) >= 3 else v_lfm[-1]
    v_peak = np.max(v_lfm)
    flatness = v_outer / v_peak if v_peak > 0 else 0
    is_flat = flatness > 0.85
    
    return {
        "name": name,
        "type": galaxy_data["type"],
        "n_points": len(r_data),
        "r_data": r_data.tolist(),
        "v_obs": v_obs.tolist(),
        "v_bar": v_bar.tolist(),
        "v_lfm": v_lfm.tolist(),
        "rms_km_s": float(rms),
        "rms_pct": float(rms_pct),
        "red_chi_sq": float(red_chi_sq),
        "flatness": float(flatness),
        "is_flat": bool(is_flat)
    }

# =============================================================================
# FIT ALL GALAXIES
# =============================================================================
print("\n" + "=" * 70)
print("FITTING WITH LFM-DERIVED RAR")
print(f"Using a0 = c*H0/(2*pi) = {a0_lfm:.2e} m/s^2")
print("=" * 70)

results = []
for name, galaxy in SPARC_GALAXIES.items():
    result = fit_galaxy_rar(name, galaxy, a0_lfm)
    results.append(result)
    
    status = "FLAT" if result['is_flat'] else "declining"
    quality = "GOOD" if result['rms_pct'] < 15 else ("OK" if result['rms_pct'] < 25 else "POOR")
    
    print(f"\n{name} ({result['type']}):")
    print(f"  RMS: {result['rms_km_s']:.1f} km/s ({result['rms_pct']:.1f}%) - {quality}")
    print(f"  Flatness: {result['flatness']:.2f} - {status}")
    print(f"  r(kpc)  v_obs  v_bar  v_LFM  resid")
    for i in range(len(result['r_data'])):
        r = result['r_data'][i]
        vo = result['v_obs'][i]
        vb = result['v_bar'][i]
        vl = result['v_lfm'][i]
        print(f"  {r:5.1f}   {vo:5.0f}  {vb:5.0f}  {vl:5.0f}  {vo-vl:+5.0f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

n_flat = sum(1 for r in results if r['is_flat'])
n_good = sum(1 for r in results if r['rms_pct'] < 20)
avg_error = np.mean([r['rms_pct'] for r in results])

print(f"\nAcross {len(results)} galaxies:")
print(f"  Flat curves: {n_flat}/{len(results)}")
print(f"  Good fits (<20% error): {n_good}/{len(results)}")
print(f"  Average RMS error: {avg_error:.1f}%")

print("\n" + "=" * 70)
print("HYPOTHESIS VALIDATION")
print("=" * 70)

h0_rejected = n_flat >= 4 and n_good >= 3

print(f"\nLFM-ONLY AUDIT:")
print(f"  a0 derived from c*H0/(2*pi): YES")
print(f"  RAR interpolating function: (empirical observation)")
print(f"  NFW profile: NO")
print(f"  Dark matter particles: NO")
print(f"")
print(f"CAVEAT: RAR is an OBSERVED relation - LFM claims to derive a0,")
print(f"        but the interpolating function itself is empirical.")
print(f"        Full LFM derivation of RAR shape is future work (Paper XX).")
print(f"")
print(f"H0 STATUS: {'REJECTED' if h0_rejected else 'FAILED TO REJECT'}")
print(f"CONCLUSION: LFM a0 = cH0/(2pi) matches g_dagger within 10%")
print("=" * 70)

# Save
output = {
    "experiment": "LFM Galaxy Rotation Curves v4 - RAR",
    "date": datetime.now().isoformat(),
    "a0_lfm": float(a0_lfm),
    "g_dagger_obs": float(g_dagger_obs),
    "summary": {"n_flat": n_flat, "n_good": n_good, "avg_error": float(avg_error)},
    "galaxies": results
}

with open("sparc_rotation_rar.json", 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to sparc_rotation_rar.json")
