# LFM Galaxy Rotation Curve Experiments

## Overview

This directory contains experiments validating the **LFM Radial Acceleration Relation (LFM-RAR)** against observed galaxy rotation curves using public SPARC data.

## The LFM-RAR Formula

```
g_obs² = g_bar² + g_bar × a₀

where:
  g_obs = observed gravitational acceleration
  g_bar = Newtonian (baryonic) acceleration
  a₀ = c × H₀ / (2π) = 1.08 × 10⁻¹⁰ m/s²
```

Equivalently: **g_obs = g_bar × √(1 + a₀/g_bar)**

This is a **geometric mean** formula - simpler than all empirical MOND interpolating functions and derived from LFM cosmological boundary conditions.

## Key Results

| Galaxy | RMS Error | Flatness | Notes |
|--------|-----------|----------|-------|
| NGC6503 | 11.0% | 0.94 | Classic spiral |
| NGC2403 | 7.4% | 0.99 | Extended HI disk |
| NGC3198 | 11.4% | 0.99 | Prototype flat curve |
| NGC7331 | 21.0% | 0.91 | Massive spiral |
| UGC128 | 12.5% | 0.99 | Low surface brightness |
| **AVERAGE** | **12.7%** | **0.96** | **5/5 flat** |

## Data Source

**SPARC (Spitzer Photometry & Accurate Rotation Curves) Database**

- Website: http://astroweb.case.edu/SPARC/
- Reference: Lelli, McGaugh & Schombert (2016), AJ, 152, 157
- License: Public data for research use
- DOI: 10.3847/0004-6256/152/6/157

The rotation curve data embedded in the Python scripts is extracted from the SPARC database. The original database contains 175 galaxies with:
- 3.6 μm Spitzer photometry (baryonic mass)
- High-quality HI/Hα rotation curves
- Decomposed mass models (gas + stellar disk + bulge)

## Files

### Main Experiment
- **`lfm_sparc_final.py`** - Final LFM-RAR fit to 5 SPARC galaxies (12.7% avg RMS)
- **`ROTATION_CURVE_RESULTS.md`** - Detailed results and interpretation

### Supporting Experiments
- `lfm_sparc_rotation.py` - Initial rotation curve experiment
- `lfm_sparc_rotation_v2.py` - V2 with improved fit
- `lfm_sparc_rotation_v3.py` - V3 with flatness analysis
- `lfm_sparc_rotation_rar.py` - RAR formula implementation

### Derivation Attempts
- `lfm_rar_derivation_attempt.py` - Initial derivation from GOV-01/02
- `lfm_rar_derivation_v2.py` - Enhanced derivation
- `lfm_rar_derivation_v3.py` - Full physical interpretation
- `lfm_rar_derivation_final.py` - Final derivation with product structure

### GOV-03/04 Experiments
- `lfm_gov03_memory.py` - Chi memory effects
- `lfm_gov03_scan.py` - Parameter scanning
- `lfm_gov04_proper.py` - Poisson limit comparison
- `lfm_dynamic_tau.py` - Dynamic tau parameter
- `lfm_dynamic_tau_v2.py` - Enhanced dynamic tau

### Assessment
- `HONEST_ASSESSMENT.md` - Honest evaluation of what works and what doesn't

## Physical Interpretation

The LFM-RAR emerges from the acceleration formula:

```
g = c × (1/χ) × (dχ/dr)
```

Where:
- The gradient dχ/dr includes both **local** (Newtonian) and **cosmological** (a₀) contributions
- The factor 1/χ depends on the local chi well depth
- The **product structure** creates the geometric mean behavior

### Limiting Cases

| Regime | Condition | Behavior |
|--------|-----------|----------|
| **Newtonian** | g_bar >> a₀ | g_obs → g_bar |
| **Deep MOND** | g_bar << a₀ | g_obs → √(g_bar × a₀) |
| **Transition** | g_bar ≈ a₀ | g_obs ≈ √2 × g_bar |

## How to Run

```bash
python lfm_sparc_final.py
```

Expected output:
```
LFM SPARC ROTATION CURVE FIT
g_obs = sqrt(g_bar^2 + g_bar * a_0)
...
AVERAGE RMS: 12.7%
FLAT CURVES: 5/5
H0 STATUS: REJECTED
```

## Citation

If using this work, please cite:

1. **SPARC Data**: Lelli, McGaugh & Schombert (2016), AJ, 152, 157
2. **LFM Framework**: Partin (2025), LFM Paper Series

## License

Experiment code: MIT License
SPARC data: Used under research fair use; see original source for terms.
