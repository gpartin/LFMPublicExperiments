# Quantum Mechanics Experiments

Experiments demonstrating quantum mechanics emerging from LFM wave dynamics.

## Physics

In LFM, quantum mechanics IS wave mechanics:
- "Particles" are standing waves in χ-wells
- Energy quantization comes from boundary conditions
- Uncertainty principle comes from wave packet properties

## Experiments

### `lfm_particle_in_box.py`

**What it tests**: Discrete energy levels in a bounded χ-well.

**Mechanism**:
- High χ at walls creates reflecting boundaries
- Only wavelengths λ_n = 2L/n fit in the box
- From GOV-01 dispersion ω² = c²k² + χ₀², we get discrete ω_n

**Results**:
- 4 discrete modes detected (n = 1, 3, 5, 7)
- 89.9% of power concentrated in discrete peaks
- Mode frequencies match theory within 5%
- H₀ REJECTED: Quantization EMERGES from GOV-01

**Why only odd modes?**: The off-center initial condition projects 
primarily onto odd modes (which have non-zero amplitude at the 
starting position).

## Running

```bash
python lfm_particle_in_box.py
```

Output saved to `particle_in_box_results.json`.

## The Deep Connection

The GOV-01 equation:
```
∂²E/∂t² = c²∇²E − χ²E
```

Is the **Klein-Gordon equation**. In the non-relativistic limit, 
it reduces to the Schrödinger equation. So quantum mechanics is 
ALREADY in LFM - we just needed to recognize it.
