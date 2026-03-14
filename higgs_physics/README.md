# Higgs Self-Coupling from Lattice Geometry

**Prediction**: λ = 4/31 ≈ 0.129032 — the Higgs boson self-coupling derived purely from coordination shell geometry of a discrete spacetime lattice.

**Key formula**: λ = D_st / (2·D_st² − 1), where D_st is the number of spacetime dimensions.
For D_st = 4: λ = 4/31 ≈ 0.129032 (SM measured: 0.1291 ± 0.05, error 0.27%).

## Paper

The full manuscript is included in this folder:

- **[paper_075_contest_submission.pdf](paper_075_contest_submission.pdf)** — 17-page paper with derivation, figures, validation, and falsification criteria
- `paper_075_contest_submission.tex` — LaTeX source
- `generate_figures.py` — Generates all three paper figures from scratch
- `fig_*.pdf` — Pre-built figures (coordination shells, derivation chain, Mexican hat potential)

## Scripts

| Script | What it does | Requirements |
|--------|-------------|--------------|
| `reproduce_lambda_derivation.py` | **START HERE.** Reproduces every algebraic claim. | NumPy only |
| `verify_lambda.py` | High-precision verification using Python `Decimal` (100-digit). | Standard library |
| `mexican_hat_gov02_experiment.py` | Tests the Mexican hat V(χ) = λ(χ²−χ₀²)² in GOV-02 dynamics. | NumPy |
| `test_z2_universality.py` | Verifies λ = D_st/(2D_st²−1) holds for D_st = 2, 3, 4, 5. | NumPy |
| `ergodicity_scaling_test.py` | Equipartition V_q/K ratio at 16³ and 32³ (finite-size scaling). | NumPy |
| `stability_boundary_experiment.py` | Tests λ = 4/31 as the resonance/stability boundary. | NumPy |

## Quick Start

```bash
# Verify all claims (< 1 second, no GPU needed)
python reproduce_lambda_derivation.py

# High-precision check
python verify_lambda.py

# Run dynamics experiments (a few minutes each)
python mexican_hat_gov02_experiment.py
python test_z2_universality.py
python ergodicity_scaling_test.py
```

## The Derivation (Summary)

1. **Axiom 1**: Discrete 3D lattice with nearest-neighbor interactions
2. **Axiom 2**: Rotating bound states exist → angular momentum quantization
3. χ₀ = 3³ − 2³ = 19 (from discrete Laplacian mode counting)
4. The second coordination shell has z₂ = 2D_st² = 32 sites
5. Each quartic χ⁴ vertex couples to z₂ − 1 = 31 physical neighbor channels
6. **λ = D_st / (z₂ − 1) = 4/31** (vertex strength = dimensions / channels)

## Falsification

- **Strong falsification**: HL-LHC measures λ/λ_SM outside [0.95, 1.05] → LFM falsified
- **Weak falsification**: λ/λ_SM outside [0.8, 1.2] → framework survives, formula fails
- **Timeline**: HL-LHC di-Higgs measurements expected 2028–2030
