"""
Cosmic Web Filament Data Access
================================

This script demonstrates how to access public cosmic web data
for LFM experiments.

Sources:
1. IllustrisTNG - Simulations with full 3D density fields
2. SDSS - Real galaxy positions
3. Pre-computed filament catalogs

Author: LFM Research
Date: 2026-02-09
"""

import numpy as np

# =============================================================================
# OPTION 1: IllustrisTNG API (Simulations)
# =============================================================================

def get_tng_data_example():
    """
    Example of accessing IllustrisTNG data via API.
    
    First, register at: https://www.tng-project.org/users/register/
    Then get your API key from your profile.
    """
    
    print("="*60)
    print("OPTION 1: IllustrisTNG Simulation Data")
    print("="*60)
    
    example_code = '''
# Install: pip install requests h5py

import requests
import h5py
import numpy as np

# Your API key (get from https://www.tng-project.org/users/profile/)
headers = {"api-key": "YOUR_API_KEY_HERE"}

# Get a snapshot from TNG100
baseUrl = "https://www.tng-project.org/api/TNG100-1/snapshots/99/"

# Get subhalo (galaxy) catalog
subhalos_url = baseUrl + "subhalos/"
response = requests.get(subhalos_url, headers=headers)
subhalos = response.json()

# Download positions of first 1000 galaxies
params = {"limit": 1000}
r = requests.get(subhalos_url, headers=headers, params=params)
data = r.json()

# Each subhalo has:
# - 'pos': 3D position [x, y, z] in kpc/h
# - 'mass': total mass
# - 'sfr': star formation rate
# etc.

# For filament analysis, you want the density field:
# Download a cutout of the density field around a region
'''
    print(example_code)
    print("\nData available:")
    print("  - Galaxy positions (millions)")
    print("  - Gas density fields (for χ computation!)")
    print("  - Dark matter density")
    print("  - Velocities, temperatures, etc.")
    print("\nFilament extraction already done in some catalogs!")
    

# =============================================================================
# OPTION 2: SDSS Galaxy Catalog (Real Observations)
# =============================================================================

def get_sdss_data_example():
    """
    Example of accessing SDSS galaxy data.
    """
    
    print("\n" + "="*60)
    print("OPTION 2: SDSS Real Galaxy Data")
    print("="*60)
    
    example_code = '''
# Install: pip install astroquery

from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy import units as u

# Query galaxies in a region
pos = coords.SkyCoord('0h8m05.63s +14d50m23.3s', frame='icrs')

# Get spectroscopic galaxies (have redshifts = 3D positions)
result = SDSS.query_region(pos, radius=5*u.deg, 
                           spectro=True,
                           photoobj_fields=['ra', 'dec', 'z'])

# Convert to 3D positions
# z (redshift) -> distance via Hubble law
# d = c * z / H0 (for small z)
H0 = 70  # km/s/Mpc
c = 3e5  # km/s
distances = c * result['z'] / H0  # Mpc

# Now you have 3D galaxy positions!
x = distances * np.cos(np.radians(result['dec'])) * np.cos(np.radians(result['ra']))
y = distances * np.cos(np.radians(result['dec'])) * np.sin(np.radians(result['ra']))
z = distances * np.sin(np.radians(result['dec']))
'''
    print(example_code)
    print("\nData available:")
    print("  - ~2 million galaxies with spectra")
    print("  - Full 3D positions from redshifts")
    print("  - Covers 1/4 of the sky")
    print("  - Real observations (not simulations)!")


# =============================================================================
# OPTION 3: Pre-computed Filament Catalogs
# =============================================================================

def get_filament_catalogs():
    """
    Pre-computed filament catalogs from the literature.
    """
    
    print("\n" + "="*60)
    print("OPTION 3: Pre-computed Filament Catalogs")
    print("="*60)
    
    catalogs = """
1. SDSS DR7 Filament Catalog (Tempel et al. 2014)
   URL: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/438/3465
   Contains: 15,421 filaments from SDSS galaxies
   Format: Galaxy IDs, filament membership, spine coordinates

2. DisPerSE Cosmic Web (Sousbie 2011)
   URL: http://www2.iap.fr/users/sousbie/web/html/indexd41d.html
   Contains: Filament spines, critical points, voids
   Software: Can run on any density field!

3. NEXUS+ Catalog
   URL: https://github.com/cautun/NEXUS-plus
   Contains: Void/wall/filament/cluster classification
   Can apply to: TNG, SDSS, any simulation

4. Cosmic Flows (Tully et al.)
   URL: https://www.ipnl.in2p3.fr/projet/cosmicflows/
   Contains: Local Universe (within 400 Mpc)
   Includes: Laniakea supercluster, local filaments
   **THIS INCLUDES OUR LOCAL NEIGHBORHOOD!**
"""
    print(catalogs)


# =============================================================================
# OPTION 4: Simple Local Universe Data
# =============================================================================

def get_local_universe():
    """
    Data specifically about our local cosmic neighborhood.
    """
    
    print("\n" + "="*60)
    print("OPTION 4: Local Universe (Nearest Filaments to Earth)")
    print("="*60)
    
    local_data = """
The Milky Way sits in a specific location in the cosmic web:

LOCAL STRUCTURES:
-----------------
1. Local Group (our cluster)
   - Milky Way + Andromeda + ~80 smaller galaxies
   - Size: ~10 million light years
   
2. Local Sheet
   - Flat structure containing Local Group
   - Part of the Virgo Supercluster
   
3. Local Filament
   - Connects Local Group to Virgo Cluster
   - Length: ~50 million light years
   - Contains: Canes Venatici, NGC groups
   
4. Virgo Cluster (nearest major cluster)
   - Distance: ~54 million light years
   - ~1500 galaxies
   - Center of mass for our supercluster

5. Laniakea Supercluster
   - Our "home" supercluster
   - Contains: 100,000 galaxies
   - Size: 520 million light years
   - Virgo is near center

PUBLICLY AVAILABLE:
-------------------
Cosmic Flows-3 catalog:
https://www.ipnl.in2p3.fr/projet/cosmicflows/

Contains:
- Distances to 18,000 galaxies within 400 Mpc
- Peculiar velocities (flow toward attractors)
- 3D density reconstruction
- **Filament map of local universe!**
"""
    print(local_data)


# =============================================================================
# LFM EXPERIMENT: Compute χ from Density
# =============================================================================

def lfm_filament_experiment_plan():
    """
    Plan for LFM experiment using filament data.
    """
    
    print("\n" + "="*60)
    print("LFM EXPERIMENT PLAN: χ-Field in Cosmic Filaments")
    print("="*60)
    
    plan = """
HYPOTHESIS:
-----------
In LFM, χ is lower where matter density is higher (E² sources χ wells).
Filaments should have LOWER χ than voids.

EXPERIMENT:
-----------
1. Get density field from TNG simulation
2. Apply GOV-02 to compute equilibrium χ:
   ∇²χ = (κ/c²)(ρ - ρ_mean)
3. Compare χ along filaments vs in voids
4. Predict: Light travels faster through filaments?

MEASUREMENTS:
-------------
1. χ_filament / χ_void ratio
2. Travel time difference for light
3. Gravitational lensing differences
4. Correlation with observed large-scale structure

CODE OUTLINE:
-------------
```python
# Load TNG density field
density = load_tng_density("TNG100", snapshot=99)

# Compute χ from GOV-02 (equilibrium)
chi0 = 19.0
kappa = 1/63
rho_mean = np.mean(density)

# Solve Poisson equation for χ
# ∇²(χ - χ0) = (kappa/c²)(density - rho_mean)
from scipy.ndimage import laplace
chi_deviation = solve_poisson(kappa * (density - rho_mean))
chi = chi0 + chi_deviation

# Load filament catalog
filaments = load_disperse_filaments()

# Measure χ along filaments vs voids
chi_in_filaments = sample_along_spines(chi, filaments)
chi_in_voids = sample_in_voids(chi, filaments)

print(f"χ in filaments: {np.mean(chi_in_filaments):.2f}")
print(f"χ in voids: {np.mean(chi_in_voids):.2f}")
print(f"Ratio: {np.mean(chi_in_filaments)/np.mean(chi_in_voids):.4f}")
```

EXPECTED RESULTS:
-----------------
If LFM is correct:
- χ_filament < χ_void (lower χ where more matter)
- Ratio should be ~0.99-0.999 (small but measurable effect)
- Light travel time slightly faster in filaments

OBSERVABLE PREDICTIONS:
-----------------------
1. Photons from distant galaxies along filaments arrive slightly early
2. Gravitational lensing is stronger along filaments
3. CMB photons that traversed more filaments have different properties
"""
    print(plan)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("COSMIC WEB DATA ACCESS FOR LFM EXPERIMENTS")
    print("="*60)
    print("\nFour options for getting filament data:\n")
    
    get_tng_data_example()
    get_sdss_data_example()
    get_filament_catalogs()
    get_local_universe()
    lfm_filament_experiment_plan()
    
    print("\n" + "="*60)
    print("RECOMMENDED STARTING POINT")
    print("="*60)
    print("""
For LFM experiments, I recommend:

1. START WITH: IllustrisTNG
   - Full 3D density fields (can compute χ directly!)
   - Filaments already identified
   - Free API access after registration
   - JupyterLab in browser - no downloads needed!

2. THEN TRY: Cosmic Flows-3
   - Real observations of our local neighborhood
   - Includes our local filament to Virgo
   - Test LFM predictions on actual data

3. ADVANCED: Run DisPerSE on TNG
   - Extract filament spines
   - Compute χ along vs perpendicular to filaments
   - Make LFM-specific predictions

NEXT STEP: Register at https://www.tng-project.org/users/register/
""")


if __name__ == "__main__":
    main()
