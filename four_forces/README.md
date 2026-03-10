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

## Observation: Shared-Source Coupling and Centroid Artifacts

In LFM, gravity and EM share a common source: |Ψ\_total|².
For two same-colour solitons with relative phase Δθ:

```
|Ψ₁ + Ψ₂|² = |Ψ₁|² + |Ψ₂|² + 2·cos(Δθ)·|Ψ₁|·|Ψ₂|
```

The cross-term simultaneously sources gravity (GOV-02) and creates EM
energy gradients (GOV-01). Constructive interference (same-phase) deposits
`4·ψ_m²` at the midpoint; destructive (opposite-phase) deposits `0`.

This |Ψ|² difference is real and **is** the Coulomb mechanism —
interference changes the energy landscape.

### Self-Correction (κ-sweep, 2026-03-10)

An earlier version of this code claimed "like charges converge faster",
based on raw centroid measurements (same: 0.99 cells, opp: 0.66 cells).
A parameter sweep over κ and separation d disproved this:

- **At κ = 0 (no gravity)**, same-phase centroids still "converge" (+0.19 cells)
  because constructive interference pulls |Ψ|² weight toward the midpoint,
  shifting the centroid inward.
- **At κ = 0, opposite-phase centroids "diverge"** (−0.60 cells) because
  destructive interference pushes weight outward.
- After subtracting this baseline, **opposite-phase pairs get MORE
  gravitational convergence** — consistent with two separate χ-wells
  having a steeper inter-well gradient.

The apparent convergence ratio was a **centroid measurement artifact**,
not a gravitational enhancement. The EM midpoint |Ψ|² difference is
genuine, but it does not accelerate convergence.

### Measured Values (A=5, σ=3, d=10)

| Quantity | Value |
|----------|-------|
| ψ\_m = A·exp(−d²/8σ²) | 1.247 |
| Same-phase centroid shift (raw) | 0.987 cells |
| Opposite-phase centroid shift (raw) | 0.656 cells |
| \|Ψ\|² midpoint ratio | 15.4× |
| κ=0 baseline (same) | +0.19 cells (static interference bias) |
| κ=0 baseline (opp) | −0.60 cells (static interference bias) |

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
