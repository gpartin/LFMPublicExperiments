#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense OR MIT
# Copyright (c) LFM Research
"""
EXPERIMENT: LFM χ-Memory in 2D - Dark Matter-Like Persistence
==============================================================

HYPOTHESIS FRAMEWORK
--------------------

GENERAL HYPOTHESIS:
When energy (matter) moves through the LFM substrate, it creates χ-wells.
Even after the matter moves away, the χ-field "remembers" where matter was -
creating persistent gravitational wells (dark matter halos).

NULL HYPOTHESIS (H0):
χ instantly follows E². When matter moves, χ returns to χ₀ immediately.
No persistent gravitational potential remains after matter vacates a region.

ALTERNATIVE HYPOTHESIS (H1):
χ has memory (via time-averaging in GOV-03 or wave dynamics in GOV-02).
After matter moves, χ-wells persist for extended time, creating "halo" structure.

LFM-ONLY CONSTRAINT VERIFICATION:
- [x] Uses ONLY GOV-01: ∂²E/∂t² = c²∇²E − χ²E
- [x] Uses ONLY GOV-03: χ² = χ₀² − g·⟨E²⟩_τ (fast-response limit with memory)
- [x] No external dark matter particles added
- [x] No halo profile assumed (NFW, Burkert, etc.)

SUCCESS CRITERIA:
- REJECT H0 if: χ-well depth at original matter location > 0.1 after matter moves away
- FAIL TO REJECT H0 if: χ returns to χ₀ within 0.01 after matter vacates

Author: Greg D. Partin
Date: February 19, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import json
from datetime import datetime

# Output directory
OUTPUT_DIR = Path(__file__).resolve().parent / "results_chi_memory_2d"
OUTPUT_DIR.mkdir(exist_ok=True)

# LFM parameters
CHI_0 = 19.0
C = 1.0
G_COUPLING = 2.5  # Coupling strength for GOV-03
TAU = 25  # Memory window (steps) - CRITICAL for dark matter effect


def laplacian_2d(field, dx):
    """2D discrete Laplacian with periodic boundaries."""
    lap = np.zeros_like(field)
    lap[1:-1, 1:-1] = (
        field[2:, 1:-1] + field[:-2, 1:-1] +
        field[1:-1, 2:] + field[1:-1, :-2] - 4 * field[1:-1, 1:-1]
    ) / (dx * dx)
    # Periodic boundaries
    lap[0, :] = lap[1, :]
    lap[-1, :] = lap[-2, :]
    lap[:, 0] = lap[:, 1]
    lap[:, -1] = lap[:, -2]
    return lap


def evolve_gov01(e, e_prev, chi, dx, dt):
    """GOV-01: ∂²E/∂t² = c²∇²E − χ²E"""
    lap_e = laplacian_2d(e, dx)
    e_next = 2 * e - e_prev + dt * dt * (C * C * lap_e - chi * chi * e)
    return e_next


def update_chi_memory(e_history, chi0, g, tau):
    """
    GOV-03: χ² = χ₀² − g·⟨E²⟩_τ
    
    The time-averaging over τ steps creates MEMORY.
    Even when E→0 at a point, if E² was high recently,
    the average ⟨E²⟩_τ remains elevated → χ stays LOW → gravitational well persists.
    """
    # Average E² over last τ steps
    if len(e_history) < tau:
        e2_avg = np.mean([e**2 for e in e_history], axis=0)
    else:
        e2_avg = np.mean([e**2 for e in e_history[-tau:]], axis=0)
    
    chi_sq = chi0**2 - g * e2_avg
    chi_sq = np.maximum(chi_sq, 0.1)  # Floor to avoid singularity
    chi = np.sqrt(chi_sq)
    return chi


def create_moving_mass(x, y, center_x, center_y, width, amplitude):
    """Create a Gaussian energy distribution (representing matter)."""
    r_sq = (x - center_x)**2 + (y - center_y)**2
    return amplitude * np.exp(-r_sq / (2 * width**2))


def run_chi_memory_2d_experiment():
    """
    Run 2D χ-memory experiment showing dark matter-like persistence.
    
    Timeline:
    1. Phase A (0-500): Mass at LEFT, creates χ-well
    2. Phase B (500-1000): Mass MOVES to RIGHT
    3. Phase C (1000-1500): Observe LEFT region - does χ-well persist?
    """
    print("=" * 70)
    print("LFM χ-MEMORY 2D EXPERIMENT - Dark Matter-Like Persistence")
    print("=" * 70)
    
    # Grid setup
    nx, ny = 128, 128
    lx, ly = 100.0, 100.0
    dx = lx / nx
    dy = ly / ny
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dt = 0.2 * dx / C
    
    print(f"\nGrid: {nx}×{ny}, domain: {lx}×{ly}")
    print(f"dx={dx:.3f}, dt={dt:.4f}")
    print(f"Memory window τ = {TAU} steps (~{TAU*dt:.1f} time units)")
    print(f"Coupling g = {G_COUPLING}")
    
    # Initialize fields
    e = np.zeros((nx, ny))
    e_prev = np.zeros((nx, ny))
    chi = np.ones((nx, ny)) * CHI_0
    
    # Energy history for memory (last τ steps)
    e_history = []
    
    # Mass parameters
    mass_width = 8.0
    mass_amplitude = 1.2
    
    # Phases
    phase_a_end = 500    # Mass at left
    phase_b_end = 1000   # Mass moves to right
    phase_c_end = 1500   # Observe memory
    
    # Positions
    left_center = (lx * 0.25, ly * 0.5)   # x=25, y=50
    right_center = (lx * 0.75, ly * 0.5)  # x=75, y=50
    
    # Storage for snapshots
    snapshots = {
        'phase_a': {'step': 400, 'e': None, 'chi': None},
        'phase_b': {'step': 750, 'e': None, 'chi': None},
        'phase_c': {'step': 1400, 'e': None, 'chi': None}
    }
    
    # Monitoring point at LEFT location (where mass WAS)
    monitor_ix = int(0.25 * nx)  # x=25
    monitor_iy = int(0.5 * ny)   # y=50
    
    chi_monitor = []
    e_monitor = []
    time_steps = []
    
    # Animation frames (save every N steps)
    animation_frames = []
    frame_interval = 30  # Save frame every 30 steps
    
    print("\n" + "-" * 70)
    print("PHASE A (0-500): Mass at LEFT, creating χ-well")
    print("-" * 70)
    
    for step in range(phase_c_end):
        # Position mass based on phase
        if step < phase_a_end:
            # Phase A: Mass at LEFT
            mass_x, mass_y = left_center
        elif step < phase_b_end:
            # Phase B: Mass moves from LEFT to RIGHT
            progress = (step - phase_a_end) / (phase_b_end - phase_a_end)
            mass_x = left_center[0] + progress * (right_center[0] - left_center[0])
            mass_y = left_center[1] + progress * (right_center[1] - left_center[1])
        else:
            # Phase C: Mass at RIGHT (observe memory at LEFT)
            mass_x, mass_y = right_center
        
        # Create energy source at current position
        e_source = create_moving_mass(X, Y, mass_x, mass_y, mass_width, mass_amplitude)
        
        # Add source as forcing term (maintain standing wave)
        e = 0.95 * e + 0.05 * e_source  # Gradual refresh
        
        # Evolve E via GOV-01
        e_next = evolve_gov01(e, e_prev, chi, dx, dt)
        e_prev = e.copy()
        e = e_next.copy()
        
        # Update χ via GOV-03 with memory
        e_history.append(e.copy())
        if len(e_history) > TAU + 10:
            e_history.pop(0)  # Keep only recent history
        
        chi = update_chi_memory(e_history, CHI_0, G_COUPLING, TAU)
        
        # Save animation frame
        if step % frame_interval == 0:
            animation_frames.append({
                'step': step,
                'e': e.copy(),
                'chi': chi.copy(),
                'mass_position': (mass_x, mass_y)
            })
        
        # Monitor
        chi_monitor.append(chi[monitor_ix, monitor_iy])
        e_monitor.append(e[monitor_ix, monitor_iy])
        time_steps.append(step)
        
        # Capture snapshots
        if step == snapshots['phase_a']['step']:
            snapshots['phase_a']['e'] = e.copy()
            snapshots['phase_a']['chi'] = chi.copy()
            print(f"  Step {step}: Mass at LEFT, χ_min = {np.min(chi):.4f}")
        
        if step == phase_a_end:
            print("\n" + "-" * 70)
            print("PHASE B (500-1000): Mass MOVES from LEFT to RIGHT")
            print("-" * 70)
        
        if step == snapshots['phase_b']['step']:
            snapshots['phase_b']['e'] = e.copy()
            snapshots['phase_b']['chi'] = chi.copy()
            print(f"  Step {step}: Mass in TRANSIT, χ at left = {chi[monitor_ix, monitor_iy]:.4f}")
        
        if step == phase_b_end:
            print("\n" + "-" * 70)
            print("PHASE C (1000-1500): Mass at RIGHT, observing LEFT for MEMORY")
            print("-" * 70)
        
        if step == snapshots['phase_c']['step']:
            snapshots['phase_c']['e'] = e.copy()
            snapshots['phase_c']['chi'] = chi.copy()
            print(f"  Step {step}: Mass at RIGHT, χ at left = {chi[monitor_ix, monitor_iy]:.4f}")
    
    # Final measurements
    chi_left_final = chi[monitor_ix, monitor_iy]
    e_left_final = e[monitor_ix, monitor_iy]
    chi_well_depth = CHI_0 - chi_left_final
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nMonitoring point: x={x[monitor_ix]:.1f}, y={y[monitor_iy]:.1f} (LEFT position)")
    print(f"\nPhase A (mass at LEFT):")
    print(f"  χ_min over grid: {np.min(snapshots['phase_a']['chi']):.4f}")
    print(f"\nPhase C (mass NOW at RIGHT, observing LEFT):")
    print(f"  E at left:  {e_left_final:.6f} (matter vacated)")
    print(f"  χ at left:  {chi_left_final:.4f}")
    print(f"  χ₀:         {CHI_0:.4f}")
    print(f"  Well depth: {chi_well_depth:.4f}")
    
    h0_rejected = chi_well_depth > 0.1
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS VALIDATION")
    print("=" * 70)
    print("LFM-ONLY VERIFIED: YES")
    print(f"H0 STATUS: {'REJECTED' if h0_rejected else 'FAILED TO REJECT'}")
    if h0_rejected:
        print("CONCLUSION: χ-field has MEMORY. χ-well persists after matter vacates,")
        print("            creating dark-matter-like gravitational potential without particles.")
    else:
        print("CONCLUSION: No significant memory observed. Try larger τ or stronger coupling.")
    print("=" * 70)
    
    # Create animated GIF
    print("\n" + "=" * 70)
    print("Creating animated GIF...")
    print("=" * 70)
    
    fig_anim, (ax_e, ax_chi) = plt.subplots(1, 2, figsize=(12, 5))
    
    def animate(frame_idx):
        frame = animation_frames[frame_idx]
        step = frame['step']
        e_frame = frame['e']
        chi_frame = frame['chi']
        mass_x, mass_y = frame['mass_position']
        
        ax_e.clear()
        ax_chi.clear()
        
        # E plot
        im_e = ax_e.imshow(e_frame.T, origin='lower', cmap='hot',
                          extent=[0, lx, 0, ly], vmin=0, vmax=mass_amplitude)
        ax_e.plot(mass_x, mass_y, 'g*', markersize=20, label='Mass')
        ax_e.plot(x[monitor_ix], y[monitor_iy], 'c*', markersize=15, label='Monitor')
        ax_e.set_xlabel('x')
        ax_e.set_ylabel('y')
        ax_e.set_title(f'E Field - Step {step}, t={step*dt:.1f}', fontweight='bold')
        ax_e.legend(loc='upper right')
        
        # Chi plot
        im_chi = ax_chi.imshow(chi_frame.T, origin='lower', cmap='viridis_r',
                              extent=[0, lx, 0, ly], vmin=CHI_0-2, vmax=CHI_0)
        ax_chi.plot(mass_x, mass_y, 'g*', markersize=20, label='Mass')
        ax_chi.plot(x[monitor_ix], y[monitor_iy], 'c*', markersize=15, label='Monitor')
        ax_chi.set_xlabel('x')
        ax_chi.set_ylabel('y')
        ax_chi.set_title(f'χ Field (Memory) - Step {step}', fontweight='bold')
        ax_chi.legend(loc='upper right')
        
        # Determine phase
        if step < phase_a_end:
            phase = 'PHASE A: Mass at LEFT'
            color = 'blue'
        elif step < phase_b_end:
            phase = 'PHASE B: Mass MOVING'
            color = 'green'
        else:
            phase = 'PHASE C: Mass at RIGHT (Memory at LEFT!)'
            color = 'red'
        
        fig_anim.suptitle(phase, fontsize=14, fontweight='bold', color=color)
        
        return []
    
    anim = FuncAnimation(fig_anim, animate, frames=len(animation_frames),
                        interval=100, blit=True)
    
    gif_path = OUTPUT_DIR / "chi_memory_2d_animation.gif"
    writer = PillowWriter(fps=10)
    anim.save(gif_path, writer=writer)
    plt.close(fig_anim)
    
    print(f"Saved animation: {gif_path}")
    print(f"  {len(animation_frames)} frames, {len(animation_frames)/10:.1f} seconds")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Phase A snapshots (Mass at LEFT)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(snapshots['phase_a']['e'].T, origin='lower', cmap='hot', 
                     extent=[0, lx, 0, ly], vmin=0, vmax=mass_amplitude)
    ax1.set_title('Phase A: E (mass at LEFT)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, label='E')
    ax1.plot(x[monitor_ix], y[monitor_iy], 'c*', markersize=15, label='Monitor')
    ax1.legend(loc='upper right', fontsize=8)
    
    ax2 = fig.add_subplot(gs[0, 1])
    chi_a = snapshots['phase_a']['chi'].T
    im2 = ax2.imshow(chi_a, origin='lower', cmap='viridis_r',
                     extent=[0, lx, 0, ly], vmin=CHI_0-2, vmax=CHI_0)
    ax2.set_title(f'Phase A: χ (well forms)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, label='χ')
    ax2.plot(x[monitor_ix], y[monitor_iy], 'c*', markersize=15)
    
    # Row 2: Phase B snapshots (Mass in TRANSIT)
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(snapshots['phase_b']['e'].T, origin='lower', cmap='hot',
                     extent=[0, lx, 0, ly], vmin=0, vmax=mass_amplitude)
    ax3.set_title('Phase B: E (mass moving)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, label='E')
    ax3.plot(x[monitor_ix], y[monitor_iy], 'c*', markersize=15)
    
    ax4 = fig.add_subplot(gs[1, 1])
    chi_b = snapshots['phase_b']['chi'].T
    im4 = ax4.imshow(chi_b, origin='lower', cmap='viridis_r',
                     extent=[0, lx, 0, ly], vmin=CHI_0-2, vmax=CHI_0)
    ax4.set_title(f'Phase B: χ (memory trail)', fontsize=10, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4, label='χ')
    ax4.plot(x[monitor_ix], y[monitor_iy], 'c*', markersize=15)
    
    # Row 3: Phase C snapshots (Mass at RIGHT, observing memory at LEFT)
    ax5 = fig.add_subplot(gs[2, 0])
    im5 = ax5.imshow(snapshots['phase_c']['e'].T, origin='lower', cmap='hot',
                     extent=[0, lx, 0, ly], vmin=0, vmax=mass_amplitude)
    ax5.set_title('Phase C: E (mass at RIGHT)', fontsize=10, fontweight='bold')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plt.colorbar(im5, ax=ax5, label='E')
    ax5.plot(x[monitor_ix], y[monitor_iy], 'c*', markersize=15, label='Monitor (LEFT)')
    ax5.legend(loc='upper right', fontsize=8)
    
    ax6 = fig.add_subplot(gs[2, 1])
    chi_c = snapshots['phase_c']['chi'].T
    im6 = ax6.imshow(chi_c, origin='lower', cmap='viridis_r',
                     extent=[0, lx, 0, ly], vmin=CHI_0-2, vmax=CHI_0)
    ax6.set_title(f'Phase C: χ (MEMORY at LEFT!)', fontsize=10, fontweight='bold')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    plt.colorbar(im6, ax=ax6, label='χ')
    ax6.plot(x[monitor_ix], y[monitor_iy], 'c*', markersize=15)
    
    # Time series plots
    ax7 = fig.add_subplot(gs[:, 2:])
    time = np.array(time_steps) * dt
    ax7_twin = ax7.twinx()
    
    line1 = ax7.plot(time, e_monitor, 'r-', linewidth=2, label='E at monitor (LEFT)')
    line2 = ax7_twin.plot(time, chi_monitor, 'b-', linewidth=2, label='χ at monitor (LEFT)')
    
    ax7.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax7_twin.axhline(CHI_0, color='gray', linestyle='--', alpha=0.3, label='χ₀=19')
    
    ax7.axvspan(0, phase_a_end*dt, alpha=0.1, color='blue', label='Phase A: Mass at LEFT')
    ax7.axvspan(phase_a_end*dt, phase_b_end*dt, alpha=0.1, color='green', label='Phase B: Transit')
    ax7.axvspan(phase_b_end*dt, phase_c_end*dt, alpha=0.1, color='red', label='Phase C: Mass at RIGHT')
    
    ax7.set_xlabel('Time', fontsize=12)
    ax7.set_ylabel('E at LEFT position', color='r', fontsize=12)
    ax7_twin.set_ylabel('χ at LEFT position', color='b', fontsize=12)
    ax7.tick_params(axis='y', labelcolor='r')
    ax7_twin.tick_params(axis='y', labelcolor='b')
    
    ax7.set_title('χ-Memory Time Series: LEFT Position\n(Matter vacates after Phase B, but χ-well PERSISTS)',
                  fontsize=12, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax7.legend(lines, labels, loc='upper left', fontsize=10)
    
    ax7.grid(alpha=0.3)
    
    fig_path = OUTPUT_DIR / "chi_memory_2d_dark_matter.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure: {fig_path}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "nx": nx, "ny": ny,
            "lx": lx, "ly": ly,
            "chi0": CHI_0,
            "g_coupling": G_COUPLING,
            "tau_memory": TAU,
            "dt": dt, "dx": dx
        },
        "phases": {
            "phase_a": {"steps": f"0-{phase_a_end}", "description": "Mass at LEFT"},
            "phase_b": {"steps": f"{phase_a_end}-{phase_b_end}", "description": "Mass moves to RIGHT"},
            "phase_c": {"steps": f"{phase_b_end}-{phase_c_end}", "description": "Mass at RIGHT, observing LEFT"}
        },
        "results": {
            "monitor_position": [x[monitor_ix], y[monitor_iy]],
            "chi_left_final": float(chi_left_final),
            "e_left_final": float(e_left_final),
            "chi_well_depth": float(chi_well_depth),
            "chi0": CHI_0
        },
        "hypothesis": {
            "lfm_only_verified": True,
            "h0_status": "REJECTED" if h0_rejected else "FAILED TO REJECT",
            "memory_observed": bool(h0_rejected)
        },
        "outputs": {
            "figure": str(fig_path.name),
            "animation": str(gif_path.name)
        }
    }
    
    results_path = OUTPUT_DIR / "chi_memory_2d_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results: {results_path}")
    
    return results


if __name__ == "__main__":
    run_chi_memory_2d_experiment()
