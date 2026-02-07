# Gravity Experiments

Experiments demonstrating gravity emerging from χ-well dynamics.

## Physics

In LFM, gravity emerges from GOV-02:
```
∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)
```

Where energy concentrations (high E²) create χ-wells. Waves curve 
toward lower χ - this IS gravity.

## Experiments

### `lfm_binary_merger.py`

**What it tests**: Two χ-sources inspiraling, merging, and ringing down.

**Mechanism**:
- Two E-sources create overlapping χ-wells
- χ-gradient force causes attraction
- Sources inspiral, merge, emit ringdown oscillations

**Results**:
- Inspiral detected: separation 60 → 5.47
- Merger at t ≈ 1160
- 23 ringdown oscillations detected
- H₀ REJECTED: Merger dynamics EMERGE from GOV-01/02

### `lfm_qnm_rigorous.py`

**What it tests**: Quasi-normal mode (ringdown) oscillations of a single χ-well.

**Mechanism**:
- Perturb a χ-well away from equilibrium
- Oscillations decay as energy radiates away
- Damping rate depends on well parameters

## Running

```bash
python lfm_binary_merger.py
python lfm_qnm_rigorous.py
```

## LFM-Only Verification

- No F = GMm/r² injected
- No Schwarzschild metric assumed
- No Kerr solution imposed
- χ-wells and their dynamics computed from GOV-01/02 only
