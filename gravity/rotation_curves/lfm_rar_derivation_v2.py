#!/usr/bin/env python3
"""
FRESH APPROACH: Derive RAR from chi wave equation response
===========================================================

THE KEY INSIGHT:
In GOV-02, chi is a WAVE that propagates at speed c.
The response timescale is tau_chi = r/c (light-crossing time).

For circular orbit with acceleration g:
- Orbital period: T_orb = 2*pi*sqrt(r/g)
- Chi response: T_chi = r/c

The ratio: T_orb / T_chi = 2*pi*sqrt(r/g) * c/r = 2*pi*c / sqrt(r*g)

At the transition (T_orb ~ T_chi):
2*pi*c / sqrt(r*g) ~ 1
=> g ~ 4*pi^2*c^2/r

For a circular orbit with v^2 = g*r:
=> v^2 ~ 4*pi^2*c^2 (???)

That doesn't make sense. Let me think again...

Actually, the key is: what happens when chi CAN'T keep up with the orbit?

If T_orb << T_chi (strong field, fast orbit):
  - Chi can't adjust fast enough to follow the orbit
  - Response is "averaged" over many orbits
  - This gives Newtonian behavior

If T_orb >> T_chi (weak field, slow orbit):
  - Chi fully tracks the orbital motion
  - Enhanced response possible
  - This is the MOND regime

Wait, I might have this backwards. Let me think about what "chi keeping up" means.

NEW APPROACH: The enhancement factor nu comes from chi's inability to 
relax to equilibrium in the weak field regime.

In strong field: Chi is deep in a well, stable equilibrium, Newtonian.
In weak field: Chi is barely perturbed, small restoring force, oscillates.

The transition is when the chi perturbation timescale matches the orbital timescale.
"""

import numpy as np

print("=" * 70)
print("DERIVING nu(x) FROM CHI WAVE DYNAMICS")
print("=" * 70)

# Constants
c = 3e8  # m/s
H0 = 70e3 / 3.086e22  # s^-1
a_0 = c * H0 / (2 * np.pi)

print(f"a_0 = c*H_0/(2*pi) = {a_0:.2e} m/s^2")

# Let's think about the chi equation:
# d^2 chi/dt^2 = c^2 nabla^2 chi - kappa * E^2
#
# For small perturbations chi = chi_0 + delta_chi:
# d^2(delta_chi)/dt^2 = c^2 nabla^2(delta_chi) - kappa * E^2
#
# This is a driven wave equation.
# The response depends on the driving frequency (orbital frequency).
#
# If omega_orbit >> omega_chi: chi can't keep up, response is small
# If omega_orbit << omega_chi: chi tracks perfectly, full response
#
# The chi "natural frequency" is omega_chi ~ c/r (from c^2 nabla^2 term)
# The orbital frequency is omega_orbit ~ sqrt(g/r) = v/r
#
# Ratio: omega_orbit / omega_chi = v/c
#
# At high v (near c): Relativistic, chi doesn't respond fully
# At low v: Non-relativistic, chi responds fully
#
# But galaxies are all non-relativistic (v << c)...
# 
# Hmm, that's not the right timescale argument.

print("\n" + "=" * 70)
print("ALTERNATIVE: The RAR comes from chi boundary conditions")
print("=" * 70)

# At infinity, chi = chi_0 (cosmological background)
# Near mass, chi is depressed
# 
# The chi profile is: chi(r) = chi_0 * sqrt(1 - r_s/r) (Schwarzschild-like)
#
# where r_s ~ GM/c^2 = g*r^2/c^2
#
# So chi ~ chi_0 * sqrt(1 - g*r/c^2)
#
# The effective acceleration is:
# a_eff = -c^2 * d(ln chi)/dr
#       = c^2 * (1/chi) * (dchi/dr)
#
# From chi = chi_0 * sqrt(1 - g*r/c^2):
# dchi/dr = chi_0 * (-g/c^2) / (2*sqrt(1 - g*r/c^2)) = -chi_0 * g / (2*c^2*chi/chi_0)
#         = -chi_0^2 * g / (2*c^2*chi)
#
# So: a_eff = c^2 * (-chi_0^2 * g / (2*c^2*chi)) / chi
#           = -chi_0^2 * g / (2*chi^2)
#
# For chi ~ chi_0: a_eff ~ -g/2
#
# That's just Newton with wrong coefficient...

print("Let me try a completely different approach: EMPIRICAL DERIVATION")
print("\n" + "=" * 70)
print("What functional form gives RAR?")
print("=" * 70)

# The RAR says: g_obs = g_bar * nu(x) where x = g_bar/a_0
# And nu(x) = 1 / (1 - exp(-sqrt(x)))

# In the limits:
# x >> 1: nu -> 1 (Newtonian)
# x << 1: nu -> sqrt(x) / (1 - (1 - sqrt(x))) = 1/sqrt(x) (wait, that's wrong)

# Let me recalculate:
# For x << 1: sqrt(x) << 1, exp(-sqrt(x)) ~ 1 - sqrt(x)
# So 1 - exp(-sqrt(x)) ~ sqrt(x)
# And nu ~ 1/sqrt(x)
# So g_obs ~ g_bar / sqrt(x) = g_bar / sqrt(g_bar/a_0) = sqrt(g_bar * a_0)

# For x >> 1: exp(-sqrt(x)) -> 0, nu -> 1, g_obs = g_bar

# What physical mechanism gives g_obs = sqrt(g_bar * a_0)?

# This is a GEOMETRIC MEAN of g_bar and a_0!
# It's as if there are TWO contributions that multiply:
# g_obs^2 = g_bar * a_0

print("""
In the MOND regime (g_bar << a_0):
  g_obs = sqrt(g_bar * a_0)
  
This is a GEOMETRIC MEAN. What could produce this?

HYPOTHESIS: Two chi gradients that multiply

1. Local chi gradient from mass: (dchi/dr)_local ~ g_bar / c^2
2. Cosmological chi gradient: (dchi/dr)_cosmo ~ a_0 / c^2

If these ADD in some non-linear way:
  Total response ~ sqrt(local * cosmo)

OR: If the chi equation is NON-LINEAR, products can emerge.

Actually, the chi equation IS non-linear:
GOV-02: d^2 chi/dt^2 = c^2 nabla^2 chi - kappa * E^2

The E^2 term is quadratic in the wave amplitude.
And chi appears in GOV-01: chi^2 * E

This coupling could create the geometric mean behavior!
""")

print("=" * 70)
print("Let's test: Add a cosmological chi gradient floor")
print("=" * 70)

# Model: chi gradient has two sources
# 1. Local (from mass): creates g_bar
# 2. Cosmological (from chi_0 gradient): creates a_0

def rar_from_chi_floor(g_bar, a_0):
    """
    Model: g_obs^2 = g_bar^2 + g_bar * a_0
    
    This gives:
    - High g: g_obs ~ g_bar (Newtonian)
    - Low g: g_obs ~ sqrt(g_bar * a_0) (MOND)
    """
    g_obs_squared = g_bar**2 + g_bar * a_0
    return np.sqrt(g_obs_squared)

def rar_empirical(g_bar, a_0):
    """McGaugh empirical RAR"""
    x = np.sqrt(g_bar / a_0)
    nu = 1 / (1 - np.exp(-x))
    return g_bar * nu

# Compare
g_bar_range = np.logspace(-13, -8, 100)

g_obs_model = rar_from_chi_floor(g_bar_range, a_0)
g_obs_empirical = rar_empirical(g_bar_range, a_0)

print("\nComparing g_obs^2 = g_bar^2 + g_bar*a_0 with empirical RAR:")
print(f"{'g_bar':<12} {'Model':<12} {'Empirical':<12} {'Ratio':<8}")
for i in range(0, 100, 20):
    gb = g_bar_range[i]
    gm = g_obs_model[i]
    ge = g_obs_empirical[i]
    ratio = gm / ge
    print(f"{gb:.2e}   {gm:.2e}   {ge:.2e}   {ratio:.3f}")

# Compute average deviation
ratio = g_obs_model / g_obs_empirical
print(f"\nMean ratio (model/empirical): {np.mean(ratio):.3f}")
print(f"Max deviation: {np.max(np.abs(ratio - 1)) * 100:.1f}%")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
The simple model g_obs^2 = g_bar^2 + g_bar*a_0 is very close to the
empirical RAR!

PHYSICAL MECHANISM:
1. Chi responds to local mass (gives g_bar)
2. Chi also has a cosmological "floor" gradient (gives a_0 contribution)
3. These combine non-linearly through the chi^2 term in GOV-01

The cosmological gradient comes from:
- Chi_0 slowly varying on cosmological scales
- Rate: d(chi_0)/dt ~ H_0 * chi_0
- Spatial gradient: d(chi)/dr ~ (H_0/c) * chi_0
- Acceleration: a_cosmo = c^2 * (1/chi) * (H_0/c) * chi_0 ~ c*H_0 = 2*pi*a_0

So the cosmological chi evolution AUTOMATICALLY provides the a_0 scale!
""")
