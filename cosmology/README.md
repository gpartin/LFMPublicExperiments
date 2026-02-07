# Cosmology Experiments

Experiments demonstrating cosmological phenomena from LFM χ-dynamics.

## Physics

Large-scale χ evolution produces:
- **Dark energy**: χ-gradient energy at cosmic scales
- **χ-horizon**: Finite universe boundary where χ → ∞
- **Cosmic expansion**: Global χ decrease stretches wavelengths

## Experiments

### `lfm_chi_horizon_analysis.py`

**What it tests**: Structure of the cosmic χ-horizon.

**Mechanism**:
- At large distances, χ increases toward χ₀
- The "horizon" is where χ becomes large enough that waves can't propagate
- This creates a finite, bounded universe from LFM dynamics

**Results**:
- Horizon structure emerges from GOV-02 equilibrium
- Dark energy fraction Ω_Λ = (χ₀-6)/χ₀ = 13/19 ≈ 0.684

## Running

```bash
python lfm_chi_horizon_analysis.py
```

## The Ω_Λ Prediction

From χ₀ = 19:
```
Ω_Λ = (χ₀ - 6) / χ₀ = 13/19 = 0.6842
```

Measured: 0.685 ± 0.007

Error: 0.12%

This is the dark energy fraction of the universe, derived from 
a single integer χ₀ = 19.
