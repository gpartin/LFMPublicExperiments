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

Where:
- **E** = Wave amplitude/energy field
- **χ** = Local "mass" parameter (creates curvature)
- **χ₀ = 19** = Background χ (fundamental constant)
- **κ = 0.016** = Coupling constant

## What Emerges

| Force/Phenomenon | Mechanism | Folder |
|------------------|-----------|--------|
| **Gravity** | χ-wells from energy concentration | `gravity/` |
| **Electromagnetism** | Phase θ interference | `electromagnetism/` |
| **Strong Force** | χ-gradients between color sources | `nuclear_physics/` |
| **Weak Force** | Momentum density coupling | `nuclear_physics/` |
| **Quantum Mechanics** | Wave boundary conditions | `quantum_mechanics/` |
| **Classical Mechanics** | Wave packet dynamics | `classical_mechanics/` |
| **Cosmology** | Large-scale χ evolution | `cosmology/` |

## Repository Structure

```
LFMPublicExperiments/
├── classical_mechanics/    # Projectile motion, orbits
├── quantum_mechanics/      # Particle in box, tunneling
├── gravity/                # Kepler, precession, binary mergers
├── electromagnetism/       # Coulomb, charge dynamics
├── nuclear_physics/        # QGP, confinement, parity
├── cosmology/              # Dark energy, horizons
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
- Proton/electron mass ratio = 5χ₀² + 2χ₀ - 7 = 1836 (0.008% error)
- Strong coupling α_s = 2/(χ₀-2) = 0.1176 (0.25% error)
- Number of gluons = χ₀ - 11 = 8 (EXACT)
- Fermion generations = (χ₀-1)/6 = 3 (EXACT)
- Dark energy fraction Ω_Λ = (χ₀-6)/χ₀ = 0.684 (0.12% error)

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

CC-BY 4.0 - Attribution required
