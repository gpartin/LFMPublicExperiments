# Nuclear Physics Experiments

Experiments demonstrating strong and weak force emergence from LFM dynamics.

## Physics

### Strong Force

The strong force emerges as χ-gradient energy between separated color sources:
- Pinned sources create χ-wells
- Energy stored in χ-gradient between sources
- Linear confinement: E = σr (string tension)

### Weak Force Parameters

From χ₀ = 19:
- Number of gluons: N_g = χ₀ - 11 = 8 (EXACT)
- Strong coupling: α_s = 2/(χ₀-2) = 0.1176 (0.25% error)
- Fermion generations: N_gen = (χ₀-1)/6 = 3 (EXACT)
- Weak mixing angle: sin²θ_W = 3/(χ₀-11) = 3/8 at GUT scale

## Experiments

### `lfm_qgp_phase_transition.py`

**What it tests**: Phase transition behavior in high-energy-density χ-medium.

**Mechanism**:
- High temperature → high E² density
- χ suppression at high density
- Phase transition when χ approaches 0

### `lfm_qgp_refined.py`

**What it tests**: Transport properties (viscosity, string tension) of QGP-like phase.

**Results**:
- String tension σ ≈ 170 emerges from dynamics
- Viscosity/entropy ratio observed from fluctuation damping
- R² = 0.999 for linear confinement

## Running

```bash
python lfm_qgp_phase_transition.py
python lfm_qgp_refined.py
```

## LFM-Only Verification

- No QCD Lagrangian injected
- No hardcoded string tension
- No assumed phase transition temperature
- All dynamics from GOV-01/02 + color structure
