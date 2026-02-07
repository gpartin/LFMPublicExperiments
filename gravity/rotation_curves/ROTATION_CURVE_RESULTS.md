# LFM Rotation Curve Results
## Summary

The LFM formula g_obs = sqrt(g_bar^2 + g_bar  a_0) with _0 = c  H_0 / (2π) fits SPARC galaxy rotation curves with **12.7% average RMS error** and produces **flat curves in 5/5 galaxies**.

## The Formula

`
g_obs^2 = g_bar^2 + g_bar  a_0

where:
  g_obs = observed gravitational acceleration (from rotation curves)
  g_bar = Newtonian (baryonic) acceleration
  a_0 = c  H_0 / (2π) = 1.08  10^-10 m/s^2
`

## Physical Interpretation

The formula g = g_bar  (g_bar + a_0) is a PRODUCT of two terms:
- **g_bar**: Newtonian acceleration from local mass distribution
- **g_bar + a_0**: Newtonian + cosmological floor from chi evolution

This arises from the LFM acceleration formula g = c  (1/χ)  (dχ/dr):
- The gradient dχ/dr includes both local and cosmological contributions
- The factor 1/χ depends on the local chi well depth
- The PRODUCT structure creates the geometric mean behavior

## Limiting Cases

| Regime | Condition | Prediction |
|--------|-----------|------------|
| **Newtonian** | g_bar >> a_0 | g_obs  g_bar |
| **Deep MOND** | g_bar << a_0 | g_obs  (g_bar  a_0) |
| **Transition** | g_bar  a_0 | g_obs  2  g_bar |

## Results by Galaxy

| Galaxy | RMS Error | Flatness | g_bar/a_0 Range |
|--------|-----------|----------|-----------------|
| NGC6503 | 11.0% | 0.94 | 0.05-0.93 |
| NGC2403 | 7.4% | 0.99 | 0.06-1.52 |
| NGC3198 | 11.4% | 0.99 | 0.05-1.22 |
| NGC7331 | 21.0% | 0.91 | 0.15-4.37 |
| UGC128 | 12.5% | 0.99 | 0.01-0.22 |
| **AVERAGE** | **12.7%** | **0.96** | - |

## Comparison with Empirical RAR

The McGaugh empirical RAR uses: ν(x) = 1 / (1 - exp(-x)) where x = g_bar/a_0

Our LFM formula uses: ν(x) = (1 + 1/x)

Agreement between formulas: **<12% across 5 decades of g_bar**

## Derivation Status

 **a_0 DERIVED**: c  H_0 / (2π) from cosmological chi evolution
  - The 2π factor comes from orbital averaging of the cosmological chi gradient
  - LFM value: 1.08  10^-10 m/s matches observed g = 1.2  10^-10 m/s (10% agreement)

 **Formula MOTIVATED**: g = g_bar  (g_bar + a_0) from product structure
  - Arises from g = (gradient)/(chi) where both terms depend on local gravity
  - Gradient includes both local and cosmological contributions
  - Full rigorous derivation from GOV-01/02 still needed

## What This Means

LFM provides a **mechanism** for the RAR:
1. Cosmological chi evolution creates a floor acceleration a_0
2. Local chi wells create Newtonian gravity g_bar
3. The product structure of g = (dχ/dr)/χ combines these
4. The resulting formula fits rotation curves without dark matter particles

## Open Questions

1. Why does NGC7331 (highest mass) have the largest error (21%)?
2. What is the full derivation of the product formula from GOV-01/02?
3. Does the formula work for elliptical galaxies and clusters?
4. What about the Bullet Cluster?

## Files

- lfm_sparc_final.py - Main fit to 5 SPARC galaxies
- lfm_rar_derivation_v3.py - Derivation attempt for the formula
- lfm_rar_derivation_final.py - Physical interpretation
