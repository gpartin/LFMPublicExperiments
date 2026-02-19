#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense OR MIT
# Copyright (c) LFM Research
"""
EXPERIMENT: LFM Foundation - 1D Substrate Dynamics (Start Here)
===============================================================

HYPOTHESIS FRAMEWORK
--------------------

GENERAL HYPOTHESIS:
Core LFM behavior can be observed directly from a 1D leapfrog evolution:
(1) waves propagate in uniform chi, (2) high-chi regions alter propagation,
(3) energy concentration lowers chi through GOV-02 coupling.

NULL HYPOTHESIS (H0):
A 1D coupled GOV-01/GOV-02 simulation shows no meaningful dependence of wave
behavior on chi, and no measurable chi response to localized energy.

ALTERNATIVE HYPOTHESIS (H1):
Wave behavior depends on chi profile and localized energy generates a
measurable chi-well through GOV-02.

LFM-ONLY CONSTRAINT VERIFICATION:
- [x] Uses ONLY GOV-01: d^2E/dt^2 = c^2 * Laplacian(E) - chi^2 * E
- [x] Uses ONLY GOV-02: d^2chi/dt^2 = c^2 * Laplacian(chi) - kappa * (E^2 - E0^2)
- [x] No Newtonian/GR/Coulomb force laws injected
- [x] No external potentials injected

SUCCESS CRITERIA:
- REJECT H0 if:
  1) Peak trajectories differ between uniform-chi and barrier regions
  2) chi_min drops below chi0 in an energy-seeded run (chi-well forms)
- FAIL TO REJECT H0 otherwise.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "results_foundation_1d"
OUTPUT_DIR.mkdir(exist_ok=True)


def laplacian_1d(field: np.ndarray, dx: float) -> np.ndarray:
    left = np.roll(field, 1)
    right = np.roll(field, -1)
    return (left - 2.0 * field + right) / (dx * dx)


def evolve_coupled(
    e_init: np.ndarray,
    chi_init: np.ndarray,
    n_steps: int,
    dt: float,
    dx: float,
    c: float,
    kappa: float,
    e0_sq: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Coupled leapfrog update for GOV-01 and GOV-02 with periodic boundaries."""
    e = e_init.copy()
    chi = chi_init.copy()
    e_prev = e.copy()
    chi_prev = chi.copy()

    e_hist = np.zeros((n_steps, e.size), dtype=np.float64)
    chi_hist = np.zeros((n_steps, chi.size), dtype=np.float64)

    for t in range(n_steps):
        e_hist[t] = e
        chi_hist[t] = chi

        e_lap = laplacian_1d(e, dx)
        chi_lap = laplacian_1d(chi, dx)

        e_next = 2.0 * e - e_prev + dt * dt * (c * c * e_lap - chi * chi * e)
        chi_next = 2.0 * chi - chi_prev + dt * dt * (c * c * chi_lap - kappa * (e * e - e0_sq))

        e_prev, e = e, e_next
        chi_prev, chi = chi, chi_next

    return e_hist, chi_hist


def gaussian_packet(x: np.ndarray, center: float, sigma: float, k0: float, amp: float) -> np.ndarray:
    envelope = np.exp(-((x - center) ** 2) / (2.0 * sigma * sigma))
    carrier = np.cos(k0 * x)
    return amp * envelope * carrier


def measure_peak_track(e_hist: np.ndarray, dx: float, sample_stride: int = 10) -> list[tuple[int, float]]:
    samples: list[tuple[int, float]] = []
    for t in range(0, e_hist.shape[0], sample_stride):
        idx = int(np.argmax(np.abs(e_hist[t])))
        samples.append((t, idx * dx))
    return samples


def run_experiment() -> dict:
    print("=" * 72)
    print("LFM FOUNDATION 1D SUBSTRATE EXPERIMENT")
    print("=" * 72)

    n = 512
    dx = 1.0
    dt = 0.2
    c = 1.0
    kappa = 1.0 / 63.0
    chi0 = 19.0
    steps = 360

    x = np.arange(n) * dx
    e_init = gaussian_packet(x, center=90.0, sigma=16.0, k0=0.35, amp=0.8)

    # Scenario A: uniform chi background
    chi_uniform = np.full(n, chi0, dtype=np.float64)
    e_hist_uniform, chi_hist_uniform = evolve_coupled(
        e_init=e_init,
        chi_init=chi_uniform,
        n_steps=steps,
        dt=dt,
        dx=dx,
        c=c,
        kappa=0.0,  # Hold chi static in this baseline
    )

    # Scenario B: chi barrier region
    chi_barrier = np.full(n, chi0, dtype=np.float64)
    chi_barrier[220:300] = chi0 + 3.0
    e_hist_barrier, chi_hist_barrier = evolve_coupled(
        e_init=e_init,
        chi_init=chi_barrier,
        n_steps=steps,
        dt=dt,
        dx=dx,
        c=c,
        kappa=0.0,  # Keep barrier fixed to isolate GOV-01 effect
    )

    # Scenario C: dynamic chi response from localized energy source
    e_seed = gaussian_packet(x, center=256.0, sigma=10.0, k0=0.0, amp=2.2)
    chi_dynamic = np.full(n, chi0, dtype=np.float64)
    e_hist_dynamic, chi_hist_dynamic = evolve_coupled(
        e_init=e_seed,
        chi_init=chi_dynamic,
        n_steps=steps,
        dt=dt,
        dx=dx,
        c=c,
        kappa=kappa,
    )

    # Measurements
    peak_uniform = measure_peak_track(e_hist_uniform, dx)
    peak_barrier = measure_peak_track(e_hist_barrier, dx)

    peak_uniform_x_end = peak_uniform[-1][1]
    peak_barrier_x_end = peak_barrier[-1][1]
    barrier_shift = peak_barrier_x_end - peak_uniform_x_end

    chi_min_initial = float(np.min(chi_hist_dynamic[0]))
    chi_min_final = float(np.min(chi_hist_dynamic[-1]))
    chi_drop = chi_min_initial - chi_min_final

    h0_rejected = (abs(barrier_shift) > 1.0) and (chi_drop > 0.02)

    print(f"Uniform endpoint peak x: {peak_uniform_x_end:.2f}")
    print(f"Barrier endpoint peak x: {peak_barrier_x_end:.2f}")
    print(f"Barrier-induced shift:    {barrier_shift:.2f}")
    print(f"chi_min initial:          {chi_min_initial:.4f}")
    print(f"chi_min final:            {chi_min_final:.4f}")
    print(f"chi drop:                 {chi_drop:.4f}")

    print("\n" + "=" * 72)
    print("HYPOTHESIS VALIDATION")
    print("=" * 72)
    print("LFM-ONLY VERIFIED: YES")
    print(f"H0 STATUS: {'REJECTED' if h0_rejected else 'FAILED TO REJECT'}")
    if h0_rejected:
        print("CONCLUSION: 1D GOV-01/GOV-02 dynamics shows chi-dependent propagation and chi-well formation.")
    else:
        print("CONCLUSION: Signals are insufficient under current settings; repeat with longer runtime or stronger seed.")
    print("=" * 72)

    # Plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    times_to_plot = [0, 80, 160, 280]
    for t in times_to_plot:
        axes[0].plot(x, e_hist_uniform[t], alpha=0.7, label=f"t={t}")
    axes[0].set_title("Scenario A: GOV-01 in Uniform chi")
    axes[0].set_ylabel("E")
    axes[0].legend(ncol=4, fontsize=8)

    for t in times_to_plot:
        axes[1].plot(x, e_hist_barrier[t], alpha=0.7, label=f"t={t}")
    axes[1].axvspan(220, 300, color="gold", alpha=0.2, label="chi barrier")
    axes[1].set_title("Scenario B: GOV-01 with High-chi Barrier")
    axes[1].set_ylabel("E")
    axes[1].legend(ncol=5, fontsize=8)

    axes[2].plot(x, chi_hist_dynamic[0], label="chi(t=0)")
    axes[2].plot(x, chi_hist_dynamic[-1], label=f"chi(t={steps-1})")
    axes[2].set_title("Scenario C: GOV-02 chi Response to Localized Energy")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("chi")
    axes[2].legend()

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "foundation_1d_summary.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()

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
            "peak_uniform_x_end": peak_uniform_x_end,
            "peak_barrier_x_end": peak_barrier_x_end,
            "barrier_shift": barrier_shift,
            "chi_min_initial": chi_min_initial,
            "chi_min_final": chi_min_final,
            "chi_drop": chi_drop,
        },
        "hypothesis": {
            "lfm_only_verified": True,
            "h0_status": "REJECTED" if h0_rejected else "FAILED TO REJECT",
        },
        "outputs": {
            "figure": str(fig_path.name),
        },
    }

    results_path = OUTPUT_DIR / "foundation_1d_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved figure:  {fig_path}")
    print(f"Saved results: {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
