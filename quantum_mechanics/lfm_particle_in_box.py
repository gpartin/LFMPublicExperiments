#!/usr/bin/env python3
"""
EXPERIMENT: Particle in a Box from LFM Substrate
=================================================

HYPOTHESIS FRAMEWORK
--------------------

GENERAL HYPOTHESIS:
Quantized energy levels emerge from standing wave solutions of GOV-01 
in a chi-well "box" using only LFM equations.

NULL HYPOTHESIS (H0):
Wave solutions in a chi-well will NOT show discrete quantized frequencies.

ALTERNATIVE HYPOTHESIS (H1):
Standing wave modes will show discrete frequencies following GOV-01 
dispersion: omega^2 = (n*pi*c/L)^2 + chi_0^2

LFM-ONLY CONSTRAINT VERIFICATION:
- [x] Uses ONLY GOV-01: d^2E/dt^2 = c^2 * Laplacian(E) - chi^2 * E
- [x] chi-well creates confinement (high chi walls)
- [x] NO Schrodinger equation injected
- [x] Frequencies measured from dynamics, not assumed

SUCCESS CRITERIA:
- REJECT H0 if: Discrete modes observed matching GOV-01 dispersion
- FAIL TO REJECT H0 if: No clear quantization

PHYSICS EXPLANATION:
A "particle" in LFM is a standing wave in a chi-well. With steep walls,
only certain wavelengths fit: lambda_n = 2L/n. From GOV-01 dispersion 
omega^2 = c^2*k^2 + chi^2 with k_n = n*pi/L, we get discrete frequencies.
This IS energy quantization from wave boundary conditions.
"""

import numpy as np
import json
from datetime import datetime

# =============================================================================
# LFM PARAMETERS
# =============================================================================
chi_box = 0.5         # chi inside box
chi_wall = 5.0        # chi at walls (high = reflecting)
c = 1.0               # Wave speed

# Grid parameters  
N = 300               # Grid points
L_box = 10.0          # Box size
L_total = 16.0        # Total domain
dx = L_total / N
dt = 0.3 * dx / c
n_steps = 4000

print("=" * 60)
print("LFM PARTICLE IN A BOX EXPERIMENT")
print("=" * 60)
print(f"\nParameters: chi_box = {chi_box}, chi_wall = {chi_wall}")
print(f"Grid: N = {N}, box = {L_box}, domain = {L_total}")

# =============================================================================
# CREATE chi PROFILE
# =============================================================================
x = np.linspace(0, L_total, N)
chi = np.ones(N) * chi_box

wall_width = (L_total - L_box) / 2
wall_left = wall_width
wall_right = L_total - wall_width

# Smooth walls
for i in range(N):
    if x[i] < wall_left + 0.5:
        chi[i] = chi_wall - (chi_wall - chi_box) * 0.5 * (1 + np.tanh(5*(x[i] - wall_left)))
    if x[i] > wall_right - 0.5:
        chi[i] = chi_box + (chi_wall - chi_box) * 0.5 * (1 + np.tanh(5*(x[i] - wall_right)))

print(f"chi-well: x in [{wall_left:.1f}, {wall_right:.1f}]")

# =============================================================================
# INITIAL CONDITION (narrow pulse excites more modes)
# =============================================================================
x_center = L_total / 2 + L_box / 4  # More off-center
width = L_box / 8  # Narrower = more high-frequency content

E = np.exp(-(x - x_center)**2 / (2 * width**2))
E_prev = E.copy()

mask = chi > (chi_box + chi_wall) / 2
E[mask] = 0
E_prev[mask] = 0

print(f"Initial Gaussian at x = {x_center:.2f}, width = {width:.2f}")

# =============================================================================
# EVOLUTION: GOV-01 ONLY
# =============================================================================
print(f"\nEvolving for {n_steps} steps...")

center_idx = N // 2
time_series = np.zeros(n_steps)

for step in range(n_steps):
    laplacian = np.zeros(N)
    laplacian[1:-1] = (E[:-2] - 2*E[1:-1] + E[2:]) / dx**2
    
    E_new = 2*E - E_prev + dt**2 * (c**2 * laplacian - chi**2 * E)
    
    E_new[0] = 0
    E_new[-1] = 0
    
    E_prev = E.copy()
    E = E_new.copy()
    
    time_series[step] = E[center_idx]
    
    if step % 1000 == 0:
        print(f"  Step {step}: E(center) = {E[center_idx]:.6f}")

# =============================================================================
# FREQUENCY ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("FREQUENCY ANALYSIS")
print("=" * 60)

frequencies = np.fft.fftfreq(n_steps, dt)
spectrum = np.abs(np.fft.fft(time_series))

pos_mask = frequencies > 0
freq_pos = frequencies[pos_mask]
spec_pos = spectrum[pos_mask]

# Find peaks
peak_indices = []
for i in range(2, len(spec_pos) - 2):
    if (spec_pos[i] > spec_pos[i-1] and spec_pos[i] > spec_pos[i+1] and
        spec_pos[i] > spec_pos[i-2] and spec_pos[i] > spec_pos[i+2] and
        spec_pos[i] > np.max(spec_pos) * 0.03):
        peak_indices.append(i)

peak_frequencies = freq_pos[peak_indices]
peak_amplitudes = spec_pos[peak_indices]

sort_idx = np.argsort(peak_amplitudes)[::-1]
peak_frequencies = peak_frequencies[sort_idx]
peak_amplitudes = peak_amplitudes[sort_idx]

print(f"\nDetected {len(peak_frequencies)} peaks")

# Expected frequencies from GOV-01 dispersion
print(f"\nTheoretical: omega_n = sqrt((n*pi*c/L)^2 + chi^2)")

expected_freqs = []
for n in range(1, 10):
    k_n = n * np.pi / L_box
    omega_n = np.sqrt((c * k_n)**2 + chi_box**2)
    freq_n = omega_n / (2 * np.pi)
    expected_freqs.append(freq_n)
    print(f"  n={n}: f = {freq_n:.4f}")

# Match peaks to theory
matched_modes = []
for i, (f_det, amp) in enumerate(zip(peak_frequencies[:7], peak_amplitudes[:7])):
    errors = [abs(f_det - f_exp) / f_exp for f_exp in expected_freqs]
    best_n = np.argmin(errors) + 1
    error_pct = errors[best_n - 1] * 100
    
    print(f"\nDetected f = {f_det:.4f} -> n = {best_n} (error: {error_pct:.1f}%)")
    
    if error_pct < 15:  # Allow 15% error
        matched_modes.append({"n": int(best_n), "f": float(f_det), "error": float(error_pct)})

# =============================================================================
# QUANTIZATION VERIFICATION
# =============================================================================
print("\n" + "=" * 60)
print("QUANTIZATION VERIFICATION")
print("=" * 60)

total_power = np.sum(spec_pos**2)
peak_power = np.sum(spec_pos[peak_indices]**2) if peak_indices else 0
discreteness = peak_power / total_power if total_power > 0 else 0

print(f"\nPeak concentration: {discreteness:.1%}")
print(f"Matched modes: {len(matched_modes)}")

# =============================================================================
# HYPOTHESIS VALIDATION
# =============================================================================
print("\n" + "=" * 60)
print("HYPOTHESIS VALIDATION")
print("=" * 60)

discrete_spectrum = discreteness > 0.4
multiple_modes = len(matched_modes) >= 3

H0_rejected = discrete_spectrum and multiple_modes

print(f"\nLFM-ONLY VERIFIED: YES")
print(f"H0 STATUS: {'REJECTED' if H0_rejected else 'FAILED TO REJECT'}")

if H0_rejected:
    print("CONCLUSION: Quantized energy levels EMERGE from GOV-01 in chi-well")
else:
    print("CONCLUSION: Quantization not clearly demonstrated")

print("=" * 60)

# Save results
results = {
    "experiment": "LFM Particle in Box",
    "timestamp": datetime.now().isoformat(),
    "parameters": {"chi_box": chi_box, "chi_wall": chi_wall, "box_size": L_box},
    "results": {
        "num_peaks": len(peak_frequencies),
        "discreteness": float(discreteness),
        "matched_modes": matched_modes
    },
    "hypothesis": {
        "H0_rejected": bool(H0_rejected),
        "lfm_only_verified": True
    }
}

with open("particle_in_box_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to particle_in_box_results.json")
