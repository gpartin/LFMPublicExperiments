# LFM Public Experiments

**Experimental validation of the Lattice Field Medium (LFM) substrate theory.**

This repository contains reproducible experiments demonstrating that fundamental physics **emerges** from two simple wave equations:

## The Governing Equations

**GOV-01** (Wave dynamics):
```
∂²E/∂t² = c²∇²E − χ²E
```

**GOV-02** (χ dynamics from energy):
```
∂²χ/∂t² = c²∇²χ − κ(E² − E₀²)
```

📖 **See [LFM_EQUATIONS.md](LFM_EQUATIONS.md) for complete equation reference, numerical implementation details, and experiment rules.**

Where:
- **E** = Wave amplitude/energy field
- **χ** = Local "mass" parameter (creates curvature)
- **χ₀ = 19** = Background χ (fundamental constant)
- **κ = 1/63** = Coupling constant (derived from lattice geometry)

## What Emerges

| Force/Phenomenon | Mechanism | Folder |
|------------------|-----------|--------|
| **All Four Forces** | Single lattice, GOV-01+02 only | `four_forces/` |
| **Gravity** | χ-wells from energy concentration | `gravity/` |
| **Electromagnetism** | Phase θ interference | `electromagnetism/` |
| **Strong Force** | χ-gradients between color sources | `nuclear_physics/` |
| **Weak Force** | Momentum density coupling | `nuclear_physics/` |
| **Quantum Mechanics** | Wave boundary conditions | `quantum_mechanics/` |
| **Classical Mechanics** | Wave packet dynamics | `classical_mechanics/` |
| **Cosmology** | Large-scale χ evolution | `cosmology/` |
| **Higgs Physics** | Self-coupling from lattice geometry | `higgs_physics/` |

## Repository Structure

```
LFMPublicExperiments/
├── notebooks/                  # Colab notebooks (interactive tutorials)
├── classical_mechanics/        # Projectile motion, orbits
├── quantum_mechanics/          # Particle in box, tunneling
├── gravity/                    # Kepler, precession, binary mergers
│   ├── gravitational_waves/    #   Binary merger, NS merger, QNM ringdown
│   ├── relativistic_effects/   #   Time dilation, frame dragging, SEP
│   └── rotation_curves/        #   175 SPARC galaxy fits
├── electromagnetism/           # Coulomb, charge dynamics
├── four_forces/                # All 4 forces from GOV-01/02
├── nuclear_physics/            # QGP, confinement
│   └── qgp_phase/             #   Phase transition experiments
├── cosmology/                  # Dark energy, horizons, cosmic web
├── higgs_physics/              # Self-coupling λ=4/31 from geometry
├── tools/                      # Verification & test utilities
├── LFM_EQUATIONS.md            # Equation reference & experiment rules
└── README.md
```

## Running Experiments

Each experiment:
1. Uses **ONLY** GOV-01/02 (no external physics injected)
2. Has explicit hypothesis framework (H₀ / H₁)
3. Reports whether H₀ is REJECTED or NOT REJECTED
4. Saves results to JSON

```bash
cd classical_mechanics
python lfm_projectile_motion.py
```

## Beginner Start Here (/LFMPhysics Day 1)

Run the first physics-only substrate tutorial experiment:

- `gravity/lfm_foundation_1d_substrate.py`

This script demonstrates three core LFM behaviors from GOV-01/GOV-02 only:
1. Wave propagation in uniform χ background
2. Propagation change across a high-χ barrier
3. χ-well formation from localized energy via GOV-02 coupling

## Experiment Naming Convention

Files follow: `lfm_{phenomenon}_{detail}.py`

Examples:
- `lfm_projectile_motion.py` - Parabolic trajectory emergence
- `lfm_particle_in_box.py` - Quantized energy levels
- `lfm_binary_merger.py` - Black hole inspiral/merger/ringdown
- `lfm_qgp_phase_transition.py` - Quark-gluon plasma

## The Key Insight

**χ₀ = 19 determines all of physics.**

From this single integer:
- Fine structure constant α = (χ₀-8)/(480π) → 1/137.088 (0.04% error)
- Proton/electron mass ratio = (χ₀-8)³ + χ₀² + (χ₀-7)² = 1836 (0.008% error)
- Strong coupling α_s = 2/(χ₀-2) = 0.1176 (0.25% error)
- Number of gluons = χ₀ - 11 = 8 (EXACT)
- Fermion generations = (χ₀-1)/6 = 3 (EXACT)
- Dark energy fraction Ω_Λ = (χ₀-2D)/χ₀ = 13/19 = 0.6842 (0.12% error)

## Citation

```bibtex
@misc{partin2026lfm,
  author = {Partin, Greg D.},
  title = {LFM Public Experiments},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/gpartin/LFMPublicExperiments}
}
```

## License

MIT License — see [LICENSE](LICENSE)
