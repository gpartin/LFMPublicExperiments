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

### `lfm_frame_dragging.py` ⭐ NEW

**What it tests**: Angular momentum transfer from rotating source to test wave (Lense-Thirring effect).

**Mechanism**:
- Rotating E-source creates rotating χ-gradient pattern via GOV-02
- Test wave packet placed at distance
- Entire system evolved via coupled GOV-01/02
- Angular momentum MEASURED from wave dynamics

**Results**:
- Test wave gains L in same direction as source rotation
- ΔL_test = +100.27 (significant transfer)
- H₀ REJECTED: Frame dragging EMERGES from GOV-01/02

### `lfm_time_dilation.py` ⭐ NEW

**What it tests**: Oscillation frequency difference between waves in χ-well vs flat background.

**Mechanism**:
- Two identical wave packets, one in χ-well, one in flat χ
- Both evolved via GOV-01 with respective χ fields
- Frequency measured by FFT of central amplitude

**Results**:
- ω_flat = 2.01, ω_well = 1.21
- Frequency ratio = 0.60 (well oscillates slower)
- Time dilation factor = 1.66
- H₀ REJECTED: Time dilation EMERGES from GOV-01

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
python lfm_frame_dragging.py
python lfm_time_dilation.py
python lfm_binary_merger.py
python lfm_qnm_rigorous.py
```

## LFM-Only Verification

- No F = GMm/r² injected
- No Schwarzschild metric assumed
- No Kerr solution imposed
- χ-wells and their dynamics computed from GOV-01/02 only
