# LFM Electromagnetism Derivations

**Reference**: LFM-PAPER-045 (Sections 13.1-13.8), LFM-PAPER-065
**Last Updated**: February 10, 2026

## Overview

Electromagnetism in LFM emerges from the **phase** of complex wave fields. Electric charge is not fundamental—it is encoded in the phase angle $\theta$ of the complex wave $\Psi = |\Psi|e^{i\theta}$.

---

## 1. Governing Equation (Complex Form)

**GOV-01** with complex $\Psi \in \mathbb{C}$:

$$\frac{\partial^2 \Psi}{\partial t^2} = c^2 \nabla^2 \Psi - \chi^2 \Psi$$

Decomposing $\Psi = R \cdot e^{i\theta}$ (amplitude R, phase θ):

**Real part** (amplitude dynamics):
$$\ddot{R} - R\dot{\theta}^2 = c^2(R_{xx} - R\theta_x^2) - \chi^2 R$$

**Imaginary part** (phase/current conservation):
$$2\dot{R}\dot{\theta} + R\ddot{\theta} = c^2(2R_x\theta_x + R\theta_{xx})$$

The imaginary equation is a **continuity equation** for the probability current.

---

## 2. Charge from Phase (CALC-28 Foundation)

| Phase θ | Charge Type | Particle |
|---------|-------------|----------|
| θ = 0 | Negative | Electron |
| θ = π | Positive | Positron |

**Physical interpretation**: Charge is NOT fundamental—it's the phase angle of the wave function.

---

## 3. Coulomb Force from Interference (CALC-28)

When two wave packets overlap, total energy density:

$$|\Psi_1 + \Psi_2|^2 = |\Psi_1|^2 + |\Psi_2|^2 + 2|\Psi_1||\Psi_2|\cos(\theta_1 - \theta_2)$$

| Configuration | Phase Difference | Interference | Energy Change | Force |
|---------------|------------------|--------------|---------------|-------|
| Same charge | Δθ = 0 | Constructive | Increases | **REPEL** |
| Opposite charge | Δθ = π | Destructive | Decreases | **ATTRACT** |

**Force from energy gradient**:
$$F = -\frac{dU_{int}}{dR}$$

where the interference energy:
$$U_{int} = \int 2|\Psi_1||\Psi_2|\cos(\theta_1 - \theta_2) \, d^3x$$

---

## 4. The 1/r² Law (CALC-28 Geometry)

For Gaussian wave packets with width σ and separation R:

$$U_{int}(R) \propto \frac{1}{R} \exp\left(-\frac{R^2}{4\sigma^2}\right)$$

**Taking derivative**:
$$F = -\frac{dU_{int}}{dR} \propto \frac{1}{R^2} \left(1 + \frac{R^2}{2\sigma^2}\right)\exp\left(-\frac{R^2}{4\sigma^2}\right)$$

**At distances σ < R << ∞**: Dominant term is $1/R^2$.

**Why 1/r²?** It's geometry—the overlap integral of two 3D Gaussians falls off as the surface area of a sphere.

---

## 5. Fine Structure Constant (from χ₀ = 19)

$$\alpha = \frac{\chi_0 - 8}{480\pi} = \frac{11}{480\pi} \approx \frac{1}{137.088}$$

**Error vs measured**: 0.04%

The factor 480π = 16 × 30π arises from the full solid angle integration in 3D.

---

## 6. Magnetic Field from Current (CALC-29)

The probability current from Noether's theorem:
$$\mathbf{j} = \text{Im}(\Psi^* \nabla \Psi) = R^2 \nabla\theta$$

This current sources a magnetic-analogue field:
$$\mathbf{B} \propto \nabla \times \mathbf{j}$$

---

## 7. Lorentz Force (CALC-30)

The complete Lorentz force emerges from two mechanisms:

1. **Electric (E) term**: Phase interference energy gradients (CALC-28)
2. **Magnetic (v×B) term**: Velocity-current coupling via χ-anisotropy

$$\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

The perpendicularity of magnetic force arises because the χ-gradient induced by current j is perpendicular to both v and j.

---

## 8. Charge Quantization

Only **discrete phase values** give stable particles:
- θ = 0 (mod 2π): Stable negative charge
- θ = π (mod 2π): Stable positive charge
- θ = π/2, 3π/2: Unstable (not observed as free particles)

**Why?** Intermediate phases don't form stable interference patterns.

---

## 9. Pair Annihilation

When θ = 0 meets θ = π:
$$|e^{i \cdot 0} + e^{i\pi}|^2 = |1 + (-1)|^2 = 0$$

The wave functions cancel completely → energy radiated as photons (propagating χ-disturbances).

---

## Key Files in This Folder

| File | Description |
|------|-------------|
| `lfm_coulomb_law_demo.py` | Interactive Coulomb demonstration |
| `lfm_point_charge_coulomb_law.py` | Full simulation |
| `lfm_coulomb_verification.png` | Verification plot |

---

## References

- **LFM-PAPER-045**: Master derivation registry (CALC-28, CALC-29, CALC-30)
- **LFM-PAPER-065**: Coulomb Force from Phase Interference
- **LFM-PAPER-046**: Fine structure constant from χ₀ = 19
- Klein, O. (1926). Zeitschrift für Physik, 37, 895-906
- Gordon, W. (1926). Zeitschrift für Physik, 40, 117-133
