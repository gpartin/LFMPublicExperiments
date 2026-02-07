#!/usr/bin/env python3
"""
THE REAL QUESTION: Does LFM Predict RAR or Just Reproduce It?
=============================================================

CURRENT STATUS:
- GOV-04 alone: Gives Newtonian gravity -> Keplerian decline
- GOV-03 with tau-memory: Can produce flat curves but shape is wrong
- Borrowed RAR: Works but is empirical, not derived

THE DEEP QUESTION:
Can the RAR interpolating function be DERIVED from GOV-01 + GOV-02?

g_obs = g_bar / (1 - exp(-sqrt(g_bar / a_0)))

where a_0 = c * H_0 / (2*pi) is the LFM-predicted scale.

HYPOTHESIS:
The RAR function might emerge from the INTERACTION between:
1. Chi wave dynamics (GOV-02)
2. Chi-matter coupling (kappa)  
3. Cosmological background (chi_0 evolution)

When g_bar >> a_0: Chi responds quickly -> Newtonian
When g_bar << a_0: Chi can't keep up -> enhanced response

Let me model this transition from chi dynamics...
"""

import numpy as np

print("=" * 70)
print("DERIVING RAR FROM CHI RESPONSE TIMESCALES")
print("=" * 70)

# Constants
c = 3e8  # m/s
H0 = 70e3 / 3.086e22  # s^-1
a_0 = c * H0 / (2 * np.pi)  # LFM prediction
print(f"a_0 (LFM) = {a_0:.2e} m/s^2")

# The key insight: chi has a RELAXATION TIMESCALE
# From GOV-02: d^2 chi/dt^2 = c^2 nabla^2 chi - kappa * E^2
# In quasi-static limit, chi relaxes on timescale t_chi ~ L/c
# where L is the scale of the mass distribution

# For circular orbit with acceleration g:
# - Orbital timescale: t_orb ~ sqrt(r/g)
# - Chi relaxation: t_chi ~ r/c

# Ratio: t_orb / t_chi ~ sqrt(r/g) * c/r ~ c / sqrt(r*g)

# When g >> a_0: t_orb << t_chi, chi can't keep up, responds "linearly" -> Newtonian
# When g << a_0: t_orb >> t_chi, chi tracks matter better -> enhanced response

# Actually wait - that's backwards. Let me think again.

# In high-g regime: Strong field, chi responds quickly, full Newtonian
# In low-g regime: Weak field, chi response matters more

# The transition should happen when g ~ a_0

# Proposed mechanism:
# chi couples to E^2, but chi ALSO propagates at c
# In weak field, the chi "halo" extends further (longer memory)
# This gives enhanced effective mass at low g

print("\nCOMPARISON: Empirical RAR vs Chi-based models")
print("-" * 60)

g_bar_range = np.logspace(-12, -8, 50)  # m/s^2

# Empirical RAR (McGaugh)
def rar_empirical(g_bar, a_0):
    x = np.sqrt(g_bar / a_0)
    nu = 1 / (1 - np.exp(-x))
    return g_bar * nu

# Candidate LFM derivation 1: Simple interpolation
def rar_lfm_v1(g_bar, a_0):
    # At high g: g_obs = g_bar (Newtonian)
    # At low g: g_obs = sqrt(g_bar * a_0) (deep MOND)
    # Interpolate smoothly
    x = g_bar / a_0
    nu = (1 + np.sqrt(1 + 4/x)) / 2
    return g_bar * nu

# Candidate LFM derivation 2: Chi memory effect
def rar_lfm_v2(g_bar, a_0, alpha=1.0):
    # Memory scale tau ~ 1/sqrt(g)
    # Enhanced response when g < a_0
    x = g_bar / a_0
    # Enhancement factor from chi memory
    memory_boost = 1 + alpha / np.sqrt(x)
    return g_bar * memory_boost

g_obs_empirical = rar_empirical(g_bar_range, a_0)
g_obs_v1 = rar_lfm_v1(g_bar_range, a_0)
g_obs_v2 = rar_lfm_v2(g_bar_range, a_0, alpha=0.5)

print(f"{'g_bar (m/s^2)':<15} {'Empirical':<12} {'LFM_v1':<12} {'LFM_v2':<12}")
print("-" * 55)
for i in range(0, len(g_bar_range), 10):
    gb = g_bar_range[i]
    ge = g_obs_empirical[i]
    g1 = g_obs_v1[i]
    g2 = g_obs_v2[i]
    print(f"{gb:.2e}     {ge:.2e}   {g1:.2e}   {g2:.2e}")

# Check which matches better
print("\n" + "=" * 70)
print("MATCH QUALITY AT g_bar = a_0 (transition point)")
print("=" * 70)

idx = np.argmin(np.abs(g_bar_range - a_0))
print(f"g_bar = {g_bar_range[idx]:.2e} m/s^2 (= a_0)")
print(f"  Empirical RAR: g_obs = {g_obs_empirical[idx]:.2e} (ratio = {g_obs_empirical[idx]/g_bar_range[idx]:.2f})")
print(f"  LFM_v1:        g_obs = {g_obs_v1[idx]:.2e} (ratio = {g_obs_v1[idx]/g_bar_range[idx]:.2f})")
print(f"  LFM_v2:        g_obs = {g_obs_v2[idx]:.2e} (ratio = {g_obs_v2[idx]/g_bar_range[idx]:.2f})")

print("\n" + "=" * 70)
print("HONEST ASSESSMENT")
print("=" * 70)
print("""
WHAT LFM PROVIDES:
- a_0 = cH_0/(2*pi) - the correct scale for RAR transition

WHAT LFM DOES NOT YET PROVIDE:
- The specific functional form of RAR: g_obs/g_bar = nu(g_bar/a_0)
- GOV-03/04 don't naturally give the exp(-sqrt(x)) shape

WHAT'S NEEDED:
1. Derive nu(x) from coupled GOV-01 + GOV-02 dynamics
2. Show that chi wave propagation creates the 1/(1-exp(-sqrt(x))) form
3. This is NON-TRIVIAL and may require numerical simulation

CURRENT STATUS:
- LFM correctly predicts a_0 (10% match to observations)
- The RAR shape is BORROWED, not derived
- Deriving RAR shape from GOV equations is FUTURE WORK

This is an honest gap in the current LFM theory.
""")
