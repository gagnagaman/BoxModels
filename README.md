# Hierarchy of Ocean Box Models

**Python Translation of Mathematica Notebook:** `Work.nb` (kept locally, not included in this repository)  
**Original Author:** Halldór Björnsson

---

## 1. Overview

This directory contains a complete Python translation of the Mathematica notebook **`Work.nb`** (*"A review of a hierarchy of box models"*). The project systematically explores the analytical dynamics, Jacobian matrix formulations, eigenvalue stability spectra, and bifurcations across a hierarchy of ocean climate box models.

---

## 2. Models Translated

### A. The Stommel (1961) 2-Box Model (`stommel_model.py`)
- **Salinity ODEs:**
  $$\dot{S}_1 = \frac{1}{V_1} [-S_0 F + q(S_2 - S_1)]$$
  $$\dot{S}_2 = \frac{1}{V_2} [S_0 F + q(S_1 - S_2)]$$
- **Flow Rule:** $q = k [\beta(S_2 - S_1) - \alpha \Delta T]$.
- **Analytical Eigenvalues:**
  $$\lambda_1 = -\frac{q}{V}, \qquad \lambda_2 = -\frac{q^2 - \gamma F}{q V}$$
  Demonstrates that the upper branch ($q > \sqrt{\gamma F}$) is **linearly stable** ($\lambda_2 < 0$), while the lower branch ($q < \sqrt{\gamma F}$) is **linearly unstable** ($\lambda_2 > 0$).

---

### B. Rahmstorf's (1996) 4-Box Model — Versions I & II (`rahmstorf_4box.py`)
- **Boxes:** Box 1 (South Atlantic/Southern Ocean), Box 2 (North Atlantic deep water formation), Box 3 (Subtropical thermocline), Box 4 (Deep ocean).
- **Version I (Interhemispheric Flow Law):**
  $$q = k (\rho_2 - \rho_1) = k [\beta(S_2 - S_1) - \alpha \Theta]$$
  Evaluates the full 4×4 symbolic Jacobian matrix and its 4 eigenvalues ($\lambda_1 = 0$, $\lambda_2 = -2q$, and $\lambda_{3, 4}$). Maps the 3D eigenvalue surface and 2D stability contours across $(q, F_1, F_2)$.
- **Version II (Intra-Basin Flow Law):**
  $$q = k (\rho_2 - \rho_3) = k [\beta(S_2 - S_3) - \alpha \Theta]$$
  Shows that while the lower branch is generically unstable, **large positive values of Southern Ocean freshwater flux ($F_2$) cause the unstable region to invade the upper branch**.

---

### C. The Gnanadesikan (1999) Model (`gnanadesikan_model.py`)
- **Pycnocline Depth Equation (`AnandCubic`):**
  $$\eta \Delta\rho D^3 + \epsilon D^2 - q_\tau D - \mu = 0$$
  coupling Northern sinking $q_n$, Southern Ocean Ekman transport $q_\tau$, Gent-McWilliams eddy return flow $\epsilon D$, and low-latitude diapycnal upwelling $\mu / D$:
  $$q_n = q_\tau - \epsilon D + \frac{\mu}{D}.$$
- **Coefficients (Work.nb values, Sv-scaled):**
  $$\epsilon = \frac{A_I L_w}{L_y}\frac{1}{10^6} = 1.667\times10^{-2}\ \mathrm{Sv/m},\qquad
  \mu = \frac{K_v A}{10^6} = 3.66\times10^3\ \mathrm{Sv\,m},\qquad
  q_\tau = 36\ \mathrm{Sv}.$$
  The positive root is evaluated in the stable form
  $D = 2\mu / [\sqrt{(q_n-q_\tau)^2+4\epsilon\mu} + (q_n-q_\tau)]$.
- **Single-panel plot:**
  `gnanadesikan_f_vs_overturning.py` imports `GnanadesikanBoxModel` from
  `gnanadesikan_model.py`, so the $F$ vs. $q_n$ curve uses exactly the same
  coefficients as panel 4 of the sensitivity suite.
- **Implicit Differentiation:**
  Computes the analytical derivatives of pycnocline depth inside the Jacobian:
  $$\frac{\partial D}{\partial S_2} = -\frac{\beta \eta D^3}{-q_\tau + 2 \epsilon D + 3 \eta \Delta\rho D^2}, \qquad \frac{\partial D}{\partial S_3} = -\frac{\partial D}{\partial S_2}$$
- **Parameter Sensitivity Suite:**
  Evaluates stability and pycnocline scaling across variations in:
  - Vertical diffusivity ($K_v$)
  - Wind stress over the Drake Passage Gap ($\tau$)
  - Mesoscale eddy parameter ($A_I$)
  - Model efficiency parameter ($\eta$ / $C$)

---

## 3. Directory Contents

```
Rannsoknir/BoxModels/
├── Hierarchy_of_Box_Models.ipynb        # Interactive Jupyter Notebook mirroring Work.nb
├── stommel_model.py                     # Stommel (1961) 2-box stability analysis
├── rahmstorf_4box.py                    # Rahmstorf (1996) 4-box model (Versions I & II)
├── gnanadesikan_model.py                # Gnanadesikan (1999) 4-panel sensitivity suite
├── gnanadesikan_f_vs_overturning.py     # Single plot: F vs. Overturning equilibrium curve
├── figures/                             # Generated stability plots
│   ├── stommel_stability.png
│   ├── rahmstorf_version1_stability.png
│   ├── rahmstorf_version2_stability.png
│   ├── gnanadesikan_sensitivity_suite.png
│   └── gnanadesikan_f_vs_overturning.png
├── Work.nb                              # Original Mathematica notebook (local only)
├── .gitignore
└── README.md                            # Documentation
```

---

## 4. How to Run

### Interactive Jupyter Notebook:
```bash
cd ~/Rannsoknir/BoxModels
jupyter notebook Hierarchy_of_Box_Models.ipynb
```

### Standalone Python Scripts:
```bash
cd ~/Rannsoknir/BoxModels
python3 stommel_model.py
python3 rahmstorf_4box.py
python3 gnanadesikan_model.py
python3 gnanadesikan_f_vs_overturning.py
```

Each script writes its figures to `figures/`. The plotting methods also return
the Matplotlib figure object, so they display correctly inside the notebook.
