# LFM Governing Equations Reference (v14.3)

## The Fundamental Equations

All physics in LFM emerges from the coupled wave equations. The framework uses a **hierarchy of field representations** depending on what physics is being modeled.

### GOV-01-S: Spinor Wave Equation (Most General - Fermions)

```
(iγᵘ∂ᵤ − χ(𝐱,t))ψ = 0
```

Where ψ ∈ ℂ⁴ is a 4-component Dirac spinor. This IS the Dirac equation with spacetime-dependent mass χ.

### GOV-01-K: Klein-Gordon Wave Equation (Squared Limit - Bosons)

```
∂²Ψₐ/∂t² = c²∇²Ψₐ − χ²Ψₐ
```

Where Ψₐ ∈ ℂ, a = 1,2,3 (color components). Reduces to real E for gravity-only (Level 0).

**Discrete leapfrog form:**
```
E[t+1] = 2*E[t] - E[t-1] + dt² * (c²∇²E[t] - χ²E[t])
```

**What it describes:**
- Ψ is the wave amplitude field (energy density ~ |Ψ|²)
- Waves propagate at speed c in flat χ background
- χ acts as a mass-like term that affects oscillation frequency
- Dispersion relation: ω² = c²k² + χ²

### GOV-02: χ Wave Equation (Substrate Stiffness Field) - COMPLETE

```
∂²χ/∂t² = c²∇²χ − κ(Σₐ|Ψₐ|² + ε_W·j − E₀²) − 4λ_H·χ(χ² − χ₀²)
           └─────────────────────────────┘   └──────────────────────┘
                   Standard terms            Mexican hat self-interaction
```

Where:
- **Σₐ|Ψₐ|²** = energy density (gravity)
- **j = Im(Ψ*∇Ψ)** = momentum density (frame dragging, weak force)
- **−4λ_H·χ(χ²−χ₀²)** = Mexican hat self-interaction: V(χ) = λ_H(χ²−χ₀²)²
  - Makes χ₀ = 19 a **dynamical attractor** (not just boundary condition)
  - Z₂ symmetry: two stable vacua at +χ₀ and −χ₀
  - λ_H = 4/31 (derived from lattice geometry, see below)

**Discrete leapfrog form (full):**
```python
chi_next = 2*chi - chi_prev + dt² * (
    c²∇²chi - kappa*(E² - E0²)
    - 4*lambda_H * chi * (chi**2 - chi0**2)  # Mexican hat
)
```

**What it describes:**
- χ is the local "stiffness" of the substrate
- Energy concentrations (high E²) reduce local χ → creates χ-wells
- χ-wells ARE gravitational potential wells
- κ controls how strongly E sources χ
- **Mexican hat makes χ₀ a dynamical attractor** with V''(χ₀) = 8λ_H·χ₀²
- **BH interiors**: χ settles at −χ₀ (Z₂ second vacuum), resolving the singularity

## Parameters (ALL DERIVED FROM χ₀ = 19)

### GEO-01: χ₀ = 19 IS DERIVED

**GEO-01 (Master Formula)**:
$$\boxed{\chi_0 = 3^D - 2^D = 27 - 8 = 19}$$

χ₀ = 19 emerges from the eigenvalue structure of the 3D discrete Laplacian (see LFM-PAPER-045 THEOREM B.3).

| Symbol | Meaning | Formula | Value |
|--------|---------|---------|-------|
| **χ₀** | Background χ (flat space) | GEO-01: 3^D - 2^D | **19** (DERIVED) |
| **κ** | χ-Ψ coupling constant | 1/(4^D - 1) | **1/63 ≈ 0.01587** |
| **ε_W** | Helicity coupling | 2/(χ₀ + 1) | **0.1** |
| **λ_H** | Higgs self-coupling | D_st/(2D_st² - 1) | **4/31 ≈ 0.1290** |
| c | Wave propagation speed | | 1.0 (natural units) |
| E₀² | Background energy density | | 0 (vacuum) |

**LFM is a ZERO-PARAMETER theory**: χ₀ is derived from geometry; κ, ε_W, and λ_H all derive from χ₀.

## When to Include Each Term

| Term | Include? | Scenarios |
|------|----------|-----------|
| **−κE²** (energy) | **Always** | All simulations |
| **ε_W·j** (momentum) | When needed | Frame dragging, weak force, rotating sources |
| **−4λ_H·χ(χ²−χ₀²)** (Mexican hat) | See below | Higgs mass, BH interiors, vacuum stability |

**Mexican hat inclusion rule**: The self-interaction is always formally present but negligible when Δχ/χ₀ is small (astrophysical scales: ~10⁻⁶). **Can omit** for cosmology, rotation curves, orbital dynamics (set λ_H=0). **Must include** for Higgs mass calculations, black hole interiors (χ < 0), vacuum stability analysis, and zoomed sub-grid simulations.

**RULE**: Most simulations don't need ε_W or Mexican hat terms because χ stays close to χ₀. Include them only when modeling extreme physics (BH interiors, rotating sources, parity violation, Higgs oscillations).

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

### Mexican Hat Self-Interaction (Higgs Potential, BH Interiors)
```
− 4λ_H · χ(χ² − χ₀²)   where V(χ) = λ_H(χ² − χ₀²)²
```
- λ_H = 4/31 = D_st/(2D_st²−1) (derived from z₂ lattice geometry)
- Makes χ₀ = 19 a **dynamical attractor**: V''(χ₀) = 8λ_H·χ₀² ≈ 373
- Higgs oscillation frequency: ω_H = √(8λ_H)·χ₀ ≈ 19.30
- **Z₂ symmetry**: two stable vacua at +χ₀ and −χ₀
- BH interiors settle at −χ₀ = −19 (second vacuum, resolves singularity)
- At astrophysical scales (Δχ/χ₀ ~ 10⁻⁶), the term is negligible → existing gravity results unchanged

**When to use:** Higgs mass/self-coupling calculations, black hole interiors (χ < 0), vacuum stability analysis, early universe reheating, zoomed sub-grid simulations. For gravity-only simulations (cosmology, rotation curves), set λ_H = 0 as valid coarse-graining.

**NOTE**: The old floor term λ(−χ)³Θ(−χ) is **permanently retired** (it was ad-hoc, never derived from lattice geometry). The Mexican hat provides the same singularity protection with Z₂ symmetric minima at ±χ₀, and is derived from first principles.

### Momentum Term (Frame Dragging, Weak Force)
```
+ ε_W·j   where j = Im(Ψ*∇Ψ)
```
- j is the momentum density (Noether current from U(1))
- ε_W = 2/(χ₀+1) = 0.1 (derived from χ₀)
- Breaks parity symmetry → weak force
- Creates gravitomagnetic effects → Lense-Thirring

**When to use:** Rotating sources (frame dragging), parity violation, weak force simulations.

## Further Reading

- **LFM-PAPER-045**: Complete derivation catalog
- **LFM-PAPER-060**: Gravity emergence
- **LFM-PAPER-065**: Electromagnetism emergence
- **LFM-PAPER-066**: Nuclear force emergence

---

*Last updated: 2026-03-13 (v14.3 - Floor term RETIRED → Mexican hat self-interaction, λ_H = 4/31 derived from lattice geometry, κ = 1/(4^D-1) = 1/63)*
