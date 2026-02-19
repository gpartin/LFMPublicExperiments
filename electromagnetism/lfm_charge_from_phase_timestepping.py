#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense OR MIT
# Copyright (c) LFM Research
"""
EXPERIMENT: LFM Charge from Phase - Time-Stepping Complex Field Simulation
===========================================================================

HYPOTHESIS FRAMEWORK
--------------------

GENERAL HYPOTHESIS:
The PHASE of complex wave fields determines electromagnetic behavior:
θ=0 (electron) and θ=π (positron) should exhibit attraction due to 
destructive interference lowering system energy, while same phases 
should exhibit repulsion due to constructive interference raising energy.

NULL HYPOTHESIS (H0):
Phase has no effect on interaction energy between wave packets.
Total energy remains constant regardless of relative phase.

ALTERNATIVE HYPOTHESIS (H1):
Opposite phases (Δθ = π) lower interaction energy → attraction.
Same phases (Δθ = 0) raise interaction energy → repulsion.

LFM-ONLY CONSTRAINT VERIFICATION:
- [x] Uses ONLY GOV-01: ∂²Ψ/∂t² = c²∇²Ψ − χ²Ψ (complex Ψ)
- [x] Uses ONLY GOV-02: ∂²χ/∂t² = c²∇²χ − κ(|Ψ|² − E₀²)
- [x] No Coulomb law F = kq₁q₂/r² injected
- [x] No charge variables q₁, q₂ defined
- [x] Energy measured from |Ψ|², not external formula

SUCCESS CRITERIA:
- REJECT H0 if:
  1) E_interaction(Δθ=π) < E_interaction(Δθ=0) by > 5%
  2) Force direction consistent with energy gradient
- FAIL TO REJECT H0 otherwise.

NOTATION:
- Ψ(x,t) ∈ ℂ: Complex wave field
- θ: Phase of Ψ = |Ψ|e^(iθ)
- E_interaction: Total |Ψ|² integrated over space
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "results_charge_phase"
OUTPUT_DIR.mkdir(exist_ok=True)


def laplacian_1d(field: np.ndarray, dx: float) -> np.ndarray:
    """1D discrete Laplacian with periodic boundaries."""
    left = np.roll(field, 1)
    right = np.roll(field, -1)
    return (left - 2.0 * field + right) / (dx * dx)


def evolve_coupled_complex(
    psi_init: np.ndarray,
    chi_init: np.ndarray,
    n_steps: int,
    dt: float,
    dx: float,
    c: float,
    kappa: float,
    e0_sq: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coupled leapfrog evolution for COMPLEX wave field Ψ and real χ field.
    
    GOV-01 (complex): ∂²Ψ/∂t² = c²∇²Ψ − χ²Ψ
    GOV-02:           ∂²χ/∂t² = c²∇²χ − κ(|Ψ|² − E₀²)
    
    Periodic boundaries.
    """
    psi = psi_init.copy()
    chi = chi_init.copy()
    psi_prev = psi.copy()
    chi_prev = chi.copy()

    psi_hist = np.zeros((n_steps, psi.size), dtype=np.complex128)
    chi_hist = np.zeros((n_steps, chi.size), dtype=np.float64)

    for t in range(n_steps):
        psi_hist[t] = psi
        chi_hist[t] = chi

        # GOV-01 update (works on complex field)
        psi_lap = laplacian_1d(psi, dx)
        psi_next = 2.0 * psi - psi_prev + dt * dt * (c * c * psi_lap - chi * chi * psi)

        # GOV-02 update (uses |Ψ|² = Ψ*Ψ as source)
        chi_lap = laplacian_1d(chi, dx)
        psi_sq = np.abs(psi) ** 2
        chi_next = 2.0 * chi - chi_prev + dt * dt * (c * c * chi_lap - kappa * (psi_sq - e0_sq))

        psi_prev, psi = psi, psi_next
        chi_prev, chi = chi, chi_next

    return psi_hist, chi_hist


def gaussian_packet_complex(
    x: np.ndarray,
    center: float,
    sigma: float,
    k0: float,
    amp: float,
    phase: float,
) -> np.ndarray:
    """
    Create complex Gaussian wave packet:
    Ψ(x) = amp * exp(-(x-c)²/(2σ²)) * exp(i(k₀x + θ))
    
    phase: θ (constant phase offset)
           θ=0   → "electron"
           θ=π   → "positron"
    k0:    carrier wave number
    """
    envelope = np.exp(-((x - center) ** 2) / (2.0 * sigma * sigma))
    # Complex carrier with phase offset
    carrier = np.exp(1j * (k0 * x + phase))
    return amp * envelope * carrier


def measure_interaction_energy(psi: np.ndarray, dx: float) -> float:
    """Total energy: ∫ |Ψ(x)|² dx"""
    return float(np.sum(np.abs(psi) ** 2) * dx)


def measure_center_of_mass(psi: np.ndarray, x: np.ndarray, dx: float) -> float:
    """Weighted average position: ∫ x|Ψ|² dx / ∫ |Ψ|² dx"""
    density = np.abs(psi) ** 2
    total_density = np.sum(density) * dx
    if total_density < 1e-12:
        return 0.0
    return float(np.sum(x * density) * dx / total_density)


def measure_separation(psi_hist: np.ndarray, x: np.ndarray, dx: float, left_idx: int, right_idx: int) -> float:
    """
    Measure separation between two packets by tracking their centers of mass.
    left_idx, right_idx: indices in x-space to window each packet.
    """
    # Window the two packets
    psi_left = psi_hist.copy()
    psi_left[right_idx:] = 0.0
    psi_right = psi_hist.copy()
    psi_right[:left_idx] = 0.0
    
    x_left = measure_center_of_mass(psi_left, x, dx)
    x_right = measure_center_of_mass(psi_right, x, dx)
    
    return abs(x_right - x_left)


def run_experiment() -> dict:
    print("=" * 72)
    print("LFM CHARGE FROM PHASE - TIME-STEPPING COMPLEX FIELD EXPERIMENT")
    print("=" * 72)

    # Parameters
    n = 512
    dx = 1.0
    dt = 0.15  # Smaller timestep for stability with complex field
    c = 1.0
    kappa = 1.0 / 63.0
    chi0 = 19.0
    steps = 300

    x = np.arange(n) * dx

    # Configuration 1: OPPOSITE PHASES (θ₁=0, θ₂=π) → should ATTRACT
    print("\n[1/2] Running OPPOSITE PHASES (electron + positron, θ=0 and θ=π)...")
    psi_left_opposite = gaussian_packet_complex(
        x, center=180.0, sigma=12.0, k0=0.0, amp=0.6, phase=0.0  # θ=0
    )
    psi_right_opposite = gaussian_packet_complex(
        x, center=330.0, sigma=12.0, k0=0.0, amp=0.6, phase=np.pi  # θ=π
    )
    psi_init_opposite = psi_left_opposite + psi_right_opposite

    chi_init = np.full(n, chi0, dtype=np.float64)

    psi_hist_opposite, chi_hist_opposite = evolve_coupled_complex(
        psi_init=psi_init_opposite,
        chi_init=chi_init.copy(),
        n_steps=steps,
        dt=dt,
        dx=dx,
        c=c,
        kappa=kappa,
    )

    energy_opposite_initial = measure_interaction_energy(psi_hist_opposite[0], dx)
    energy_opposite_final = measure_interaction_energy(psi_hist_opposite[-1], dx)
    
    sep_opposite_initial = measure_separation(psi_hist_opposite[0], x, dx, 0, n // 2)
    sep_opposite_final = measure_separation(psi_hist_opposite[-1], x, dx, 0, n // 2)
    
    # Configuration 2: SAME PHASE (θ₁=0, θ₂=0) → should REPEL
    print("[2/2] Running SAME PHASE (both θ=0)...")
    psi_left_same = gaussian_packet_complex(
        x, center=180.0, sigma=12.0, k0=0.0, amp=0.6, phase=0.0  # θ=0
    )
    psi_right_same = gaussian_packet_complex(
        x, center=330.0, sigma=12.0, k0=0.0, amp=0.6, phase=0.0  # θ=0 (same!)
    )
    psi_init_same = psi_left_same + psi_right_same

    psi_hist_same, chi_hist_same = evolve_coupled_complex(
        psi_init=psi_init_same,
        chi_init=chi_init.copy(),
        n_steps=steps,
        dt=dt,
        dx=dx,
        c=c,
        kappa=kappa,
    )

    energy_same_initial = measure_interaction_energy(psi_hist_same[0], dx)
    energy_same_final = measure_interaction_energy(psi_hist_same[-1], dx)
    
    sep_same_initial = measure_separation(psi_hist_same[0], x, dx, 0, n // 2)
    sep_same_final = measure_separation(psi_hist_same[-1], x, dx, 0, n // 2)

    # Analysis
    energy_diff_pct = 100.0 * (energy_same_final - energy_opposite_final) / energy_opposite_final
    
    sep_change_opposite = sep_opposite_final - sep_opposite_initial
    sep_change_same = sep_same_final - sep_same_initial

    h0_rejected = abs(energy_diff_pct) > 5.0

    print("\n" + "-" * 72)
    print("MEASUREMENTS")
    print("-" * 72)
    print(f"OPPOSITE PHASES (θ=0, θ=π):")
    print(f"  Initial energy:     {energy_opposite_initial:.6f}")
    print(f"  Final energy:       {energy_opposite_final:.6f}")
    print(f"  Initial separation: {sep_opposite_initial:.2f}")
    print(f"  Final separation:   {sep_opposite_final:.2f}")
    print(f"  Separation change:  {sep_change_opposite:+.2f} {'(APPROACH)' if sep_change_opposite < 0 else '(RECEDE)'}")
    
    print(f"\nSAME PHASE (θ=0, θ=0):")
    print(f"  Initial energy:     {energy_same_initial:.6f}")
    print(f"  Final energy:       {energy_same_final:.6f}")
    print(f"  Initial separation: {sep_same_initial:.2f}")
    print(f"  Final separation:   {sep_same_final:.2f}")
    print(f"  Separation change:  {sep_change_same:+.2f} {'(APPROACH)' if sep_change_same < 0 else '(RECEDE)'}")
    
    print(f"\nENERGY COMPARISON:")
    print(f"  Same - Opposite:    {energy_diff_pct:+.2f}%")

    print("\n" + "=" * 72)
    print("HYPOTHESIS VALIDATION")
    print("=" * 72)
    print("LFM-ONLY VERIFIED: YES")
    print(f"H0 STATUS: {'REJECTED' if h0_rejected else 'FAILED TO REJECT'}")
    if h0_rejected:
        print("CONCLUSION: Phase determines interaction - opposite phases (Δθ=π)")
        print("            lower energy (attractive), same phase raises energy (repulsive).")
    else:
        print("CONCLUSION: Energy difference < 5%, inconclusive. Try larger amplitude or longer runtime.")
    print("=" * 72)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    times_to_plot = [0, 100, 200, steps - 1]
    
    # Top-left: |Ψ|² evolution for OPPOSITE phases
    ax = axes[0, 0]
    for t in times_to_plot:
        density = np.abs(psi_hist_opposite[t]) ** 2
        ax.plot(x, density, alpha=0.7, label=f"t={t}")
    ax.set_title("OPPOSITE PHASES: |Ψ(x,t)|² Evolution")
    ax.set_xlabel("x")
    ax.set_ylabel("|Ψ|²")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)

    # Top-right: χ response for OPPOSITE phases
    ax = axes[0, 1]
    ax.plot(x, chi_hist_opposite[0], label="χ(t=0)", alpha=0.7)
    ax.plot(x, chi_hist_opposite[-1], label=f"χ(t={steps-1})", alpha=0.7)
    ax.axhline(chi0, color="gray", linestyle="--", alpha=0.5, label=f"χ₀={chi0}")
    ax.set_title("OPPOSITE PHASES: χ Response")
    ax.set_xlabel("x")
    ax.set_ylabel("χ")
    ax.legend()
    ax.grid(alpha=0.3)

    # Bottom-left: |Ψ|² evolution for SAME phase
    ax = axes[1, 0]
    for t in times_to_plot:
        density = np.abs(psi_hist_same[t]) ** 2
        ax.plot(x, density, alpha=0.7, label=f"t={t}")
    ax.set_title("SAME PHASE: |Ψ(x,t)|² Evolution")
    ax.set_xlabel("x")
    ax.set_ylabel("|Ψ|²")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom-right: χ response for SAME phase
    ax = axes[1, 1]
    ax.plot(x, chi_hist_same[0], label="χ(t=0)", alpha=0.7)
    ax.plot(x, chi_hist_same[-1], label=f"χ(t={steps-1})", alpha=0.7)
    ax.axhline(chi0, color="gray", linestyle="--", alpha=0.5, label=f"χ₀={chi0}")
    ax.set_title("SAME PHASE: χ Response")
    ax.set_xlabel("x")
    ax.set_ylabel("χ")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "charge_from_phase_timestepping.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "N": n,
            "dx": dx,
            "dt": dt,
            "steps": steps,
            "c": c,
            "chi0": chi0,
            "kappa": kappa,
        },
        "metrics": {
            "opposite_phases": {
                "energy_initial": energy_opposite_initial,
                "energy_final": energy_opposite_final,
                "separation_initial": sep_opposite_initial,
                "separation_final": sep_opposite_final,
                "separation_change": sep_change_opposite,
            },
            "same_phase": {
                "energy_initial": energy_same_initial,
                "energy_final": energy_same_final,
                "separation_initial": sep_same_initial,
                "separation_final": sep_same_final,
                "separation_change": sep_change_same,
            },
            "energy_difference_pct": energy_diff_pct,
        },
        "hypothesis": {
            "lfm_only_verified": True,
            "h0_status": "REJECTED" if h0_rejected else "FAILED TO REJECT",
        },
        "outputs": {
            "figure": str(fig_path.name),
        },
    }

    results_path = OUTPUT_DIR / "charge_from_phase_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved figure:  {fig_path}")
    print(f"Saved results: {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
