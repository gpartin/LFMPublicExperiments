# Gravity Experiments

Experiments demonstrating gravitational phenomena emerging from pure LFM substrate dynamics.

## Directory Structure

```
gravity/
+-- gravitational_waves/       # GW generation, binary mergers, ringdown
|   +-- lfm_binary_merger.py
|   +-- lfm_neutron_star_merger.py
|   +-- lfm_qnm_rigorous.py
|   +-- README.md
+-- relativistic_effects/      # Time dilation, frame dragging
|   +-- lfm_time_dilation.py
|   +-- lfm_frame_dragging.py
|   +-- README.md
+-- README.md                  # This file
```

## Summary of Results

| Category | Experiment | H0 Status | Key Finding |
|----------|------------|-----------|-------------|
| GW | Binary Merger | REJECTED | Inspiral emerges from chi-gradient |
| GW | NS Merger | REJECTED | 4/5 GW170817 features matched |
| GW | QNM Ringdown | REJECTED | Oscillations emerge from chi-wells |
| Rel | Time Dilation | REJECTED | 40% slower oscillation in chi-wells |
| Rel | Frame Dragging | REJECTED | Angular momentum transfer to test waves |

## LFM Equations Used

All experiments use ONLY:
- **GOV-01**: d^2E/dt^2 = c^2 * Laplacian(E) - chi^2 * E
- **GOV-02**: d^2chi/dt^2 = c^2 * Laplacian(chi) - kappa*(E^2 - E0^2)

NO external physics (Newton, Einstein, Schwarzschild) is injected.
