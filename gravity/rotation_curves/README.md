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

The LFM-RAR is derived from the governing equations in 6 steps:

| Step | Status | Content |
|------|--------|---------|
| 1 | **THEOREM** | GOV-04 (quasi-static GOV-02) = Newtonian Poisson equation |
| 2 | **THEOREM** | Product structure: $g_{\text{obs}} = g_{\text{bar}} \cdot \chi_0/\chi$ from GOV-01 metric |
| 3 | **THEOREM** | $a_0 = cH_0/(2\pi) \approx 1.04 \times 10^{-10}$ m/s² from GOV-03 chi-memory |
| 4 | **THEOREM** | Newtonian limit: $g_{\text{bar}} \gg a_0 \Rightarrow g_{\text{obs}} \to g_{\text{bar}}$ |
| 5 | **PROPOSITION** | Deep-field limit: $g_{\text{bar}} \ll a_0 \Rightarrow g_{\text{obs}} \to \sqrt{g_{\text{bar}} \cdot a_0}$ |
| 6 | **THEOREM** | $f(x) = 1 + 1/x$ is the unique simplest rational interpolation |

**This is NOT borrowed from MOND.** MOND postulates an interpolation function μ(g/a₀) and fits a₀ empirically. LFM derives both the enhancement factor (from chi-field geometry) and the acceleration scale (from cosmological chi-memory timescale).

**Full derivation**: See `paper_experiments/pygrc_comparison/LFM_RAR_DERIVATION.md` in the Papers repo.

## Citation

If using this work, please cite:
- **SPARC Data**: Lelli, McGaugh & Schombert (2016), AJ, 152, 157
- **LFM Framework**: Partin (2025), LFM Paper Series
