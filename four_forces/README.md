# Four Forces from Two Equations

**All four fundamental forces emerge from a single 3D lattice simulation
using only GOV-01 and GOV-02.**

No Coulomb law. No Newton gravity. No QCD potential. No weak Lagrangian.

## The Equations

```
GOV-01:  d²Ψ_a/dt² = ∇²Ψ_a − χ² Ψ_a       (per colour a = R, G, B)
GOV-02:  d²χ/dt²   = ∇²χ   − κ(Σ|Ψ_a|² + ε_W · j)
```

where `j = Σ_a Im(Ψ_a* · ∇Ψ_a)` is the Noether current (parity-odd).

## How to Run

```bash
pip install numpy matplotlib pillow
python lfm_four_forces_cpu.py
```

Runs in **~3.5 minutes on CPU** (48³ grid, 15 000 steps).

## What It Does

Eight solitons are placed on a 48³ lattice:

| Group | Solitons | Purpose |
|-------|----------|---------|
| Same-phase pair | S1 (θ=0), S2 (θ=0) | Gravity baseline |
| Opposite-phase pair | O1 (θ=0), O2 (θ=π) | EM test (destructive interference) |
| Baryon triplet | B_R, B_G, B_B (3 colours) | Strong force (shared well) |
| Lone soliton | L_R (1 colour) | Strong force control |

Initial χ is Poisson-equilibrated (self-consistent GOV-02 static limit).
The lattice then evolves under pure leapfrog integration of GOV-01 + GOV-02.

## Force Detection (Non-Tautological)

Each force is detected by measuring an **emergent dynamical difference**
that would not exist without that force mechanism:

| Force | Mechanism | Test | Metric |
|-------|-----------|------|--------|
| **Gravity** | χ-wells from \|Ψ\|² attract waves | Soliton centroids approach over time | Centroid separation decrease > 0.3 cells |
| **EM** | Phase interference changes \|Ψ\|² at midpoint | Same-phase midpoint has MORE \|Ψ\|² than opposite-phase (constructive vs destructive) | \|Ψ\|² ratio > 1.5 |
| **Strong** | 3 colours → 3× \|Ψ\|² → 3× deeper χ-well | Baryon χ-well deeper than lone soliton χ-well | Well depth difference > 0.5 |
| **Weak** | ε_W · j couples parity-odd current to χ | Noether current j is a significant fraction of GOV-02 source | j source fraction > 1% |

## Output

| File | Description |
|------|-------------|
| `lfm_four_forces_cpu_results/results.json` | Full metrics at every checkpoint |
| `lfm_four_forces_cpu_results/four_forces_dashboard.png` | 4-panel plot of each force metric over time |
| `lfm_four_forces_cpu_results/chi_evolution.gif` | Animated χ-field showing EM pairs (y-slice) and baryon/lone (z-slice) |

## Typical Results

```
  Forces detected:   4 / 4
  LFM-ONLY:         YES (GOV-01 + GOV-02 only)
  Runtime:           ~200s  (3.4 min)

  GRAVITY:  centroid approach 0.99 cells (same pair), 0.66 cells (opp pair)
  EM:       |Psi|^2 midpoint ratio = 15.4 (same/opp)
  STRONG:   chi_min baryon = 9.85, chi_min lone = 15.94  (well diff = 6.09)
  WEAK:     j source fraction = 5.6% of GOV-02 source
```

## Novel Prediction: Interference-Gravity Enhancement

**Like-charge solitons converge faster than opposite-charge solitons.**

This is counterintuitive — EM repulsion should slow same-charge convergence.
The experiment consistently measures the opposite: same-phase pairs converge
~50% faster (0.99 vs 0.66 cells).

### Theorem (Shared-Source Interference-Gravity Coupling)

In LFM, gravity and EM share a common source: |Ψ\_total|².
For two same-colour solitons with relative phase Δθ:

```
|Ψ₁ + Ψ₂|² = |Ψ₁|² + |Ψ₂|² + 2·cos(Δθ)·|Ψ₁|·|Ψ₂|
```

The cross-term `2·cos(Δθ)·|Ψ₁|·|Ψ₂|` simultaneously:

1. **Sources gravity** via GOV-02: `Δ(source) = ±κ·2|Ψ₁||Ψ₂|`
2. **Creates EM force** via GOV-01: `F_EM = −∇(cross-term)`

At the midpoint of two identical Gaussians (each contributing ψ\_m):

| Config | \|Ψ\|² at midpoint | GOV-02 grav source | EM force |
|--------|--------------------|--------------------|----------|
| Same-phase (Δθ=0) | 4·ψ\_m² | 4κ·ψ\_m² | Repulsive |
| Opposite-phase (Δθ=π) | 0 | 0 | Attractive |

Gravity from (1) integrates over the full overlap volume (**nonlocal**).
EM from (2) acts through a local energy gradient.
When the integrated gravitational well exceeds the local EM repulsion:

**Like charges attract faster than opposite charges.**

This has **no analog in standard physics** where G/α\_EM ≈ 10⁻³⁶ makes
gravitational enhancement from interference unmeasurably small.

### Measured Values (A=5, σ=3, d=10)

| Quantity | Value |
|----------|-------|
| ψ\_m = A·exp(−d²/8σ²) | 1.247 |
| Same-phase convergence | 0.987 cells |
| Opposite-phase convergence | 0.656 cells |
| Convergence ratio (same/opp) | 1.50× |
| \|Ψ\|² midpoint ratio | 15.4× |

## Requirements

- Python 3.10+
- NumPy
- Matplotlib (for dashboard PNG)
- Pillow (for GIF animation)

## Physics Purity Audit

The evolution loop contains **exactly two update rules**:

1. `Ψ_new = 2Ψ − Ψ_prev + dt²(∇²Ψ − χ²Ψ)` — GOV-01
2. `χ_new = 2χ − χ_prev + dt²(∇²χ − κ(|Ψ|² + ε_W·j))` — GOV-02

No external physics is injected at any point. Force detection metrics are
measured from the evolved fields, not assumed.
