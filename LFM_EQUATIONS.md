# LFM Governing Equations Reference

## The Two Fundamental Equations

All physics in LFM emerges from these two coupled wave equations:

### GOV-01: E Wave Equation (Energy/Amplitude Field)

```
∂²E/∂t² = c²∇²E − χ²E
```

**Discrete leapfrog form:**
```
E[t+1] = 2*E[t] - E[t-1] + dt² * (c²∇²E[t] - χ²E[t])
```

**What it describes:**
- E is the wave amplitude field (energy density ~ E²)
- Waves propagate at speed c in flat χ background
- χ acts as a mass-like term that affects oscillation frequency
- Dispersion relation: ω² = c²k² + χ²

### GOV-02: χ Wave Equation (Substrate Stiffness Field)

```
∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)
```

**Discrete leapfrog form:**
```
χ[t+1] = 2*χ[t] - χ[t-1] + dt² * (c²∇²χ[t] - κ*(E²[t] - E₀²))
```

**What it describes:**
- χ is the local "stiffness" of the substrate
- Energy concentrations (high E²) reduce local χ → creates χ-wells
- χ-wells ARE gravitational potential wells
- κ controls how strongly E sources χ

## Parameters

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| χ₀ | Background χ (flat space) | 19 (fundamental) or normalized to 1-2 |
| κ | χ-E coupling constant | 0.016 (physical) or 0.5 (numerical) |
| c | Wave propagation speed | 1.0 (natural units) |
| E₀² | Background energy density | 0 (vacuum) |

## What Emerges

From GOV-01 + GOV-02 alone (no additional physics):

| Phenomenon | Mechanism |
|------------|-----------|
| **Gravity** | E² creates χ-wells; waves curve toward low χ |
| **Time dilation** | ω = √(c²k² + χ²); lower χ → slower oscillation |
| **Light bending** | χ-gradient refracts wave propagation |
| **Frame dragging** | Rotating E² creates rotating χ-gradient |
| **Gravitational waves** | χ disturbances propagate at c |
| **Black holes** | Deep χ-wells with event horizons |

## Numerical Implementation

### 1D Laplacian
```python
laplacian[1:-1] = (E[:-2] - 2*E[1:-1] + E[2:]) / dx**2
```

### 2D Laplacian
```python
laplacian[1:-1, 1:-1] = (
    E[:-2, 1:-1] + E[2:, 1:-1] +
    E[1:-1, :-2] + E[1:-1, 2:] -
    4 * E[1:-1, 1:-1]
) / dx**2
```

### CFL Stability Condition
```python
dt = 0.2 * dx / c  # Safe timestep
```

## Experiment Rules

All experiments in this repository MUST:

1. ✅ Use ONLY GOV-01 and GOV-02 for field evolution
2. ✅ Measure physical quantities from field values (not predict them)
3. ❌ NOT inject external physics (F = GMm/r², Coulomb, Lorentz force, etc.)
4. ❌ NOT assume metric tensors, Schwarzschild solutions, etc.
5. ❌ NOT use known formulas to calculate outcomes

Physics must EMERGE from the equations, not be inserted.

## Extensions (for specific phenomena)

### Complex Wave Field (Electromagnetism)
```
Ψ = |Ψ|e^(iθ)  where θ = phase = charge
```
- θ = 0: negative charge (electron)
- θ = π: positive charge (positron)
- Same phase → repel (constructive interference increases E²)
- Opposite phase → attract (destructive interference decreases E²)

### Multi-Component Field (Strong Force)
```
Ψₐ where a = 1, 2, 3 (color components)
E² → Σₐ|Ψₐ|²
```
- Color sources create χ-gradients between them
- Linear confinement: E = σr (string tension)

## Further Reading

- **LFM-PAPER-045**: Complete derivation catalog
- **LFM-PAPER-060**: Gravity emergence
- **LFM-PAPER-065**: Electromagnetism emergence
- **LFM-PAPER-066**: Nuclear force emergence

---

*Last updated: 2026-02-07*
