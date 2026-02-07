# Cosmology Experiments

LFM experiments probing cosmic-scale phenomena using ONLY GOV-01 and GOV-02.

## Experiments

### 1. Cosmic Expansion (lfm_cosmic_expansion_pure.py) - PURE LFM

**Challenge**: Model accelerating expansion using only substrate dynamics.

**Method** (NO Friedmann equation):
- Run GOV-01 and GOV-02 on a 1D lattice
- Start with radiation (wave) and matter (E^2 clumps)
- Let matter dilute over time (physical expansion)
- MEASURE wavelength and chi evolution
- DERIVE scale factor and Hubble parameter from measurements

**Results (H0 REJECTED)**:

| Quantity | Measured |
|----------|----------|
| Initial wavelength | 50.1 |
| Final wavelength | 501.0 |
| Expansion factor | 10x |
| Chi evolution | 19.0 -> 0.1 |
| H evolution | Decreasing |

**LFM-ONLY AUDIT**:
- Friedmann equation used: NO
- H(z) = H0*sqrt(...) used: NO
- Textbook cosmology injected: NO
- All physics from GOV-01/02: YES

**Key Finding**: Wavelength stretching (cosmic redshift) EMERGES from GOV-01/02 dynamics as matter dilutes.

### 2. Chi Horizon Analysis (lfm_chi_horizon_analysis.py)

Explores chi -> 0 as cosmic horizon boundary.

## LFM Predictions (for comparison, NOT injected)

From chi_0 = 19:
- Omega_Lambda = (chi0-6)/chi0 = 13/19 = 0.6842
- Omega_m = 6/chi0 = 6/19 = 0.3158

These are analytic predictions to be tested against full simulations.
