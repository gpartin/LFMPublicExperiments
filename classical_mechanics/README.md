# Classical Mechanics Experiments

Experiments demonstrating classical mechanics emerging from LFM wave dynamics.

## Physics

In LFM, "particles" are wave packets. Classical mechanics emerges when:
- Wave packets are localized compared to χ-gradient scale
- χ-gradients create effective potentials
- Wave packet centers of mass follow geodesics

## Experiments

### `lfm_projectile_motion.py`

**What it tests**: Parabolic trajectory of a wave packet in a χ-gradient.

**Mechanism**: 
- χ increases with height (simulates gravitational potential)
- Wave packet curves toward lower χ (toward "ground")
- Curvature produces constant acceleration → parabolic path

**Results**:
- Vertical motion R² (parabola) = 0.978
- Curves downward toward lower χ
- H₀ REJECTED: Parabolic motion EMERGES from GOV-01

## Running

```bash
python lfm_projectile_motion.py
```

Output saved to `projectile_results.json`.

## LFM-Only Verification

All experiments use ONLY:
- GOV-01: ∂²E/∂t² = c²∇²E − χ²E
- No F = ma injected
- No hardcoded trajectories
