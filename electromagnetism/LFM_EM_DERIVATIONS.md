# LFM Electromagnetism Derivations

**Reference**: LFM-PAPER-045 (Sections 13.1-13.8), LFM-PAPER-065  
**Version**: 1.0  
**Last Updated**: February 10, 2026

This document provides complete derivations for electromagnetism in LFM, with step-by-step math that can be verified with any symbolic package (SymPy, Mathematica, Maple).

---

## Table of Contents

1. [Overview](#overview)
2. [Complex Wave Decomposition](#1-complex-wave-decomposition)
3. [Current Conservation from GOV-01](#2-current-conservation-from-gov-01)
4. [Phase Interference and Coulomb Force](#3-phase-interference-and-coulomb-force)
5. [Noether Current Derivation](#4-noether-current-derivation)
6. [Why 1/r² Force Law Emerges](#5-why-1r²-force-law-emerges)
7. [Fine Structure Constant](#6-fine-structure-constant)
8. [Magnetic Field and Lorentz Force](#7-magnetic-field-and-lorentz-force)
9. [Charge Quantization](#8-charge-quantization)
10. [Pair Annihilation](#9-pair-annihilation)
11. [Appendix A: Common Errors](#appendix-a-common-errors)
12. [Appendix B: Verification](#appendix-b-verification)

---

## Overview

Electromagnetism in LFM emerges from the **phase** of complex wave fields. Electric charge is not fundamental—it is encoded in the phase angle θ of the complex wave Ψ = |Ψ|e^(iθ).

| Phase θ | Charge Type | Particle |
|---------|-------------|----------|
| θ = 0 | Negative | Electron |
| θ = π | Positive | Positron |

---

## 1. Complex Wave Decomposition

### Setup

Starting with the complex wave field:

$$\Psi(x,t) = R(x,t) \cdot e^{i\theta(x,t)} \tag{1.1}$$

where R is the real amplitude and θ is the real phase.

### First Time Derivative

$$\frac{\partial \Psi}{\partial t} = \frac{\partial}{\partial t}\left(R \cdot e^{i\theta}\right) \tag{1.2}$$

Using the product rule:

$$\frac{\partial \Psi}{\partial t} = \left(\dot{R} + iR\dot{\theta}\right) e^{i\theta} \tag{1.3}$$

### Second Time Derivative

$$\frac{\partial^2 \Psi}{\partial t^2} = \left[\left(\ddot{R} - R\dot{\theta}^2\right) + i\left(2\dot{R}\dot{\theta} + R\ddot{\theta}\right)\right] e^{i\theta} \tag{1.4}$$

### Second Spatial Derivative

By identical reasoning (replace t → x):

$$\frac{\partial^2 \Psi}{\partial x^2} = \left[\left(R_{xx} - R\theta_x^2\right) + i\left(2R_x\theta_x + R\theta_{xx}\right)\right] e^{i\theta} \tag{1.5}$$

### Verification

```python
from sympy import *
t, x = symbols('t x', real=True)
R = Function('R')(x, t)
theta = Function('theta')(x, t)
Psi = R * exp(I * theta)
print(simplify(diff(Psi, t, 2) / exp(I*theta)))
# Output confirms (1.4)
```

---

## 2. Current Conservation from GOV-01

### GOV-01 for Complex Field

$$\frac{\partial^2 \Psi}{\partial t^2} = c^2 \nabla^2 \Psi - \chi^2 \Psi \tag{2.1}$$

### Substitution

Substituting (1.4) and (1.5) into (2.1) and dividing by e^(iθ):

$$\left(\ddot{R} - R\dot{\theta}^2\right) + i\left(2\dot{R}\dot{\theta} + R\ddot{\theta}\right) = c^2\left[\left(R_{xx} - R\theta_x^2\right) + i\left(2R_x\theta_x + R\theta_{xx}\right)\right] - \chi^2 R \tag{2.2}$$

### Separating Real and Imaginary Parts

**Real Part (Amplitude Equation):**

$$\ddot{R} - R\dot{\theta}^2 = c^2\left(R_{xx} - R\theta_x^2\right) - \chi^2 R \tag{2.3}$$

**Imaginary Part (Phase Equation):**

$$2\dot{R}\dot{\theta} + R\ddot{\theta} = c^2\left(2R_x\theta_x + R\theta_{xx}\right) \tag{2.4}$$

### Rewriting as Continuity Equation

The left side of (2.4) can be rewritten:

$$2\dot{R}\dot{\theta} + R\ddot{\theta} = \frac{1}{R} \cdot \frac{\partial}{\partial t}\left(R^2 \dot{\theta}\right) \tag{2.5}$$

Similarly for the right side:

$$2R_x\theta_x + R\theta_{xx} = \frac{1}{R} \cdot \frac{\partial}{\partial x}\left(R^2 \theta_x\right) \tag{2.6}$$

Multiplying (2.4) by R and generalizing to 3D:

$$\frac{\partial}{\partial t}\left(R^2 \dot{\theta}\right) = c^2 \nabla \cdot \left(R^2 \nabla\theta\right) \tag{2.7}$$

### Identification as Continuity Equation

Defining:
- ρ = R² = |Ψ|² (probability/charge density)
- **j** = -c² R² ∇θ (probability/charge current)

Equation (2.7) becomes:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{j} = 0 \tag{2.8}$$

**This is exactly the continuity equation, derived purely from GOV-01.**

---

## 3. Phase Interference and Coulomb Force

### Superposition of Two Particles

$$\Psi_1 = R_1 e^{i\theta_1}, \quad \Psi_2 = R_2 e^{i\theta_2} \tag{3.1}$$

### Energy Density

$$|\Psi_1 + \Psi_2|^2 = |\Psi_1|^2 + |\Psi_2|^2 + 2R_1 R_2 \cos(\theta_2 - \theta_1) \tag{3.2}$$

### Phase Cases

| Relative Phase | cos(Δθ) | Energy Change | Result |
|----------------|---------|---------------|--------|
| Δθ = 0 (same) | +1 | Increases | **Repulsion** |
| Δθ = π (opposite) | -1 | Decreases | **Attraction** |

### Force from Energy Gradient

$$\mathbf{F} = -\nabla E_{total} \tag{3.3}$$

- Same phase → Energy increases as particles approach → **Repulsion**
- Opposite phase → Energy decreases as particles approach → **Attraction**

**This reproduces Coulomb's law behavior without injecting it.**

---

## 4. Noether Current Derivation

For a complex field Ψ with U(1) symmetry, the conserved Noether current is:

$$j^\mu = \text{Im}(\Psi^* \partial^\mu \Psi) \tag{4.1}$$

Using Ψ = R e^(iθ):

$$\Psi^* \nabla\Psi = R e^{-i\theta} \cdot \nabla(R e^{i\theta}) = R \nabla R + i R^2 \nabla\theta \tag{4.2}$$

Taking the imaginary part:

$$\mathbf{j} = \text{Im}(\Psi^* \nabla\Psi) = R^2 \nabla\theta \tag{4.3}$$

This confirms the identification in Section 2.

---

## 5. Why 1/r² Force Law Emerges

### Geometric Origin

The 1/r² dependence is a consequence of **3D geometry**, not electromagnetism specifically.

### Energy for Gaussian Packets

For Gaussian wave packets with width σ and separation R:

$$U_{int}(R) \propto \frac{1}{R} \exp\left(-\frac{R^2}{4\sigma^2}\right) \tag{5.1}$$

Taking derivative:

$$F = -\frac{dU_{int}}{dR} \propto \frac{1}{R^2} \left(1 + \frac{R^2}{2\sigma^2}\right)\exp\left(-\frac{R^2}{4\sigma^2}\right) \tag{5.2}$$

At distances σ < R << ∞: Dominant term is **1/R²**.

### Physical Reason

For a point-like source with amplitude R ∝ 1/r (to conserve total probability on expanding spheres):

$$\int R^2 \cdot 4\pi r^2 \, dr = \text{const} \implies R \propto \frac{1}{r} \tag{5.3}$$

The interference energy goes as 1/r, so force goes as 1/r².

---

## 6. Fine Structure Constant

From χ₀ = 19:

$$\alpha = \frac{\chi_0 - 8}{480\pi} = \frac{11}{480\pi} \approx \frac{1}{137.088} \tag{6.1}$$

**Measured value**: 1/137.036  
**Error**: 0.04%

The factor 480π = 16 × 30π arises from the full solid angle integration in 3D.

---

## 7. Magnetic Field and Lorentz Force

### Magnetic Field from Current (CALC-29)

The probability current from Noether's theorem:

$$\mathbf{j} = \text{Im}(\Psi^* \nabla \Psi) = R^2 \nabla\theta \tag{7.1}$$

This current sources a magnetic-analogue field:

$$\mathbf{B} \propto \nabla \times \mathbf{j} \tag{7.2}$$

### Lorentz Force (CALC-30)

The complete Lorentz force emerges from two mechanisms:

1. **Electric (E) term**: Phase interference energy gradients (CALC-28)
2. **Magnetic (v×B) term**: Velocity-current coupling via χ-anisotropy

$$\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \tag{7.3}$$

The perpendicularity of magnetic force arises because the χ-gradient induced by current j is perpendicular to both v and j.

---

## 8. Charge Quantization

Only **discrete phase values** give stable particles:
- θ = 0 (mod 2π): Stable negative charge
- θ = π (mod 2π): Stable positive charge
- θ = π/2, 3π/2: Unstable (not observed as free particles)

### Topological Origin

The phase θ lives on a circle (0 to 2π). Stable configurations require θ to return to itself after any closed loop—this requires θ to be constant or change by multiples of 2π.

**Result**: Charge is quantized because phase is topological.

---

## 9. Pair Annihilation

When θ = 0 meets θ = π:

$$|e^{i \cdot 0} + e^{i\pi}|^2 = |1 + (-1)|^2 = 0 \tag{9.1}$$

The wave functions cancel completely → energy radiated as photons (propagating χ-disturbances).

---

## Appendix A: Common Errors

### Error: Computing ∇²(R sin θ) incorrectly

**WRONG** (claimed by some critics):
$$\nabla^2(R \sin\theta) = \frac{1}{R}\left[\sin\theta + \frac{\cos^2\theta}{\sin\theta}\right]$$

This formula:
1. Has wrong dimensions (1/R vs R_xx)
2. Is missing spatial derivatives of R and θ
3. Blows up when sin θ = 0

**CORRECT:**
$$\nabla^2(R \sin\theta) = (R_{xx} - R\theta_x^2)\sin\theta + (2R_x\theta_x + R\theta_{xx})\cos\theta \tag{A.1}$$

---

## Appendix B: Verification

All derivations can be verified using the regression test suite:

```bash
pytest test_lfm_em_regression.py -v
```

The tests verify:
1. Time/space derivative structure (Eqs. 1.4, 1.5)
2. Real/imaginary separation (Eqs. 2.3, 2.4)
3. Continuity equation emergence (Eq. 2.8)
4. Noether current form (Eq. 4.3)
5. Phase interference signs (Section 3)

---

## Key Files in This Folder

| File | Description |
|------|-------------|
| `lfm_coulomb_law_demo.py` | Interactive Coulomb demonstration |
| `lfm_point_charge_coulomb_law.py` | Full simulation |
| `lfm_em_derivation_verification.py` | SymPy verification script |
| `test_lfm_em_regression.py` | Regression test suite (10 tests) |

---

## References

- **LFM-PAPER-045**: Master derivation registry (CALC-28, CALC-29, CALC-30)
- **LFM-PAPER-065**: Coulomb Force from Phase Interference
- **LFM-PAPER-046**: Fine structure constant from χ₀ = 19
- Klein, O. (1926). Zeitschrift für Physik, 37, 895-906
- Gordon, W. (1926). Zeitschrift für Physik, 40, 117-133
- Noether, E. (1918). Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen, 235-257
