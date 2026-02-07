# LFM Galaxy Rotation Curve Validation

## Overview

Validates the **LFM Radial Acceleration Relation (LFM-RAR)** against observed galaxy rotation curves using public SPARC data.

## The LFM-RAR Formula

```
g_obs^2 = g_bar^2 + g_bar * a_0
```

where:
- g_obs = observed gravitational acceleration
- g_bar = Newtonian (baryonic) acceleration
- a_0 = c * H_0 / (2*pi) = 1.08e-10 m/s^2 (LFM-derived)

Equivalently: **g_obs = g_bar * sqrt(1 + a_0/g_bar)**

This is a **geometric mean** formula - simpler than all empirical MOND interpolating functions.

## Results Summary

| Galaxy | RMS Error | Flatness | g_bar/a_0 Range |
|--------|-----------|----------|-----------------|
| NGC6503 | 11.0% | 0.94 | 0.05 - 0.93 |
| NGC2403 | 7.4% | 0.99 | 0.06 - 1.52 |
| NGC3198 | 11.4% | 0.99 | 0.05 - 1.22 |
| NGC7331 | 21.0% | 0.91 | 0.15 - 4.37 |
| UGC128 | 12.5% | 0.99 | 0.01 - 0.22 |
| **AVERAGE** | **12.7%** | **0.96** | - |

**Hypothesis Status**: H0 REJECTED (avg RMS 12.7% < 15% threshold)

## Data Source

**SPARC (Spitzer Photometry & Accurate Rotation Curves) Database**
- Website: http://astroweb.case.edu/SPARC/
- Citation: Lelli, McGaugh & Schombert (2016), AJ, 152, 157
- DOI: 10.3847/0004-6256/152/6/157

## Files

| File | Description |
|------|-------------|
| lfm_sparc_rotation_curves.py | Main experiment script |
| sparc_results.json | Machine-readable results |

## How to Run

```bash
python lfm_sparc_rotation_curves.py
```

## Physical Interpretation

The LFM-RAR emerges from the chi-based acceleration formula:
```
g = c^2 * (1/chi) * (d_chi/dr)
```

This has **two factors**:
1. **1/chi** - depends on local chi well depth (from baryonic mass)
2. **d_chi/dr** - gradient includes both local AND cosmological contributions

The product structure creates the geometric mean behavior naturally.

## Citation

If using this work, please cite:
- **SPARC Data**: Lelli, McGaugh & Schombert (2016), AJ, 152, 157
- **LFM Framework**: Partin (2025), LFM Paper Series
