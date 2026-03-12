# LFM: Predicting 175 Galaxy Rotation Curves from First Principles

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpartin/LFMPublicExperiments/blob/main/LFMPublicExperiments/notebooks/LFM_SPARC_Galaxy_Rotation_Curves.ipynb)

**Zero parameters fitted to rotation curves. 175 galaxies. One equation derived from wave mechanics.**

## The Result

The Lattice Field Medium (LFM) framework derives a radial acceleration relation from two coupled wave equations:

$$g_{\text{obs}} = \sqrt{g_{\text{bar}}^2 + g_{\text{bar}} \cdot a_0}, \quad a_0 = \frac{cH_0}{2\pi} = 1.08 \times 10^{-10} \text{ m/s}^2$$

Applied to the full [SPARC database](https://astroweb.cwru.edu/SPARC/) (175 late-type galaxies, 3,391 data points):

| Model | Median RMS Error | Parameters Fitted to Rotation Curves |
|-------|-----------------|--------------------------------------|
| **LFM** | **16.2%** | **0** — $a_0$ derived from cosmology |
| MOND (simple) | 18.2% | 1 — $a_0$ fitted to data |
| Newton (baryons only) | 43.3% | 0 |

Head-to-head: **LFM wins on 55% of galaxies** using an acceleration scale derived entirely from $c$ and $H_0$.

## Why This Matters

1. **MOND's $a_0$ is no longer a free parameter.** For 43 years (Milgrom 1983), the MOND acceleration scale $a_0 \approx 1.2 \times 10^{-10}$ m/s² has been an unexplained coincidence. LFM derives $a_0 = cH_0/(2\pi) = 1.08 \times 10^{-10}$ m/s² from the quasi-static limit of its governing equations — within 2% of the empirically optimal value.

2. **A testable prediction.** At the transition scale ($g_{\text{bar}} = a_0$), the LFM interpolation function gives $\nu(1) = \sqrt{2} \approx 1.414$, while MOND gives $1/(1-e^{-1}) \approx 1.582$. That's a **12% difference** measurable with current data in the transition regime.

3. **Fully reproducible.** Click the Colab badge above. The entire analysis runs in ~30 seconds with zero setup.

## The Governing Equations

LFM starts from two coupled wave equations on a discrete lattice:

- **GOV-01**: $\partial^2\Psi/\partial t^2 = c^2\nabla^2\Psi - \chi^2\Psi$ (matter wave dynamics)
- **GOV-02**: $\partial^2\chi/\partial t^2 = c^2\nabla^2\chi - \kappa(|\Psi|^2 - E_0^2)$ (substrate response)

In the quasi-static limit, $\chi$ develops wells around concentrations of energy density. Waves curve toward low-$\chi$ regions — this **is** gravity, with no external force law imposed.

The $\chi$ field is depressed by two sources: local baryonic matter (giving $g_{\text{bar}}$) and cosmological vacuum energy (giving the $a_0$ scale). Their geometric combination through the nonlinear $\chi$-$\Psi$ coupling yields the radial acceleration relation above.

## Universal Parameters

| Parameter | Value | Origin |
|-----------|-------|--------|
| $\chi_0$ | 19 | Lattice mode counting: $3^3 - 2^3$ |
| $\kappa$ | 1/63 | Non-DC modes: $1/(4^3 - 1)$ |
| $a_0$ | $1.08 \times 10^{-10}$ m/s² | Cosmological: $cH_0/(2\pi)$ |
| $\Upsilon_{\text{disk}}$ | 0.5 $M_\odot/L_\odot$ | Stellar population synthesis at 3.6μm |
| $\Upsilon_{\text{bulge}}$ | 0.7 $M_\odot/L_\odot$ | Stellar population synthesis at 3.6μm |

No parameter is adjusted per galaxy. All are either derived from the lattice structure or taken from independent astrophysical measurements.

## Repository Contents

```
notebooks/
  LFM_SPARC_Galaxy_Rotation_Curves.ipynb   # Main Colab notebook (click badge above)
gravity/                                    # χ-memory gravitational well experiments
electromagnetism/                           # Phase interference experiments
four_forces/                                # Unified four-force emergence
```

## How to Reproduce

**Option 1: Google Colab (recommended)**
Click the badge at the top. Hit "Run All." Takes ~30 seconds.

**Option 2: Local**
```bash
pip install numpy matplotlib
python -c "
import urllib.request, zipfile
urllib.request.urlretrieve('https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip', 'sparc.zip')
zipfile.ZipFile('sparc.zip').extractall('sparc_data')
"
# Then run the notebook cells locally
```

## Data Source

**SPARC** (Spitzer Photometry & Accurate Rotation Curves): Lelli, McGaugh & Schombert (2016), AJ 152, 157.
Downloaded from: https://astroweb.cwru.edu/SPARC/

## Citation

If you use this work, please cite:

```bibtex
@misc{partin2026lfm_sparc,
  author = {Partin, George},
  title = {LFM: Predicting 175 Galaxy Rotation Curves from First Principles},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/gpartin/LFMPublicExperiments}
}
```

## License

MIT License. See [LICENSE](LICENSE).
