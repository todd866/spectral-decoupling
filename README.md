# Spectral Decoupling of Dimension and Information in Network Dynamics

**Status:** Draft complete, ready for refinement

**Target Journal:** Journal of Complex Networks (OUP)

## Overview

This paper establishes that geometric dimensionality and statistical information are independent axes of network dynamical complexity:

- **Dimension (D_eff)**: Controlled by network topology via Laplacian eigenvalue density
- **Information (h_μ)**: Controlled by dynamics and noise

The core theorem proves that networks can have:
1. Matched entropy but different dimension (iso-entropy, different D)
2. Matched dimension but different entropy (iso-D, different entropy)

## Key Contribution

While the dimension-entropy distinction is classical in dynamical systems (Grassberger-Procaccia, Eckmann-Ruelle), we make it **network-native** by:

1. Defining spectral effective dimension D_eff(τ) based on Laplacian mode count
2. Proving eigenvalue density controls decoupling
3. Introducing the (D, h_μ) phase portrait as a diagnostic tool

## Figures

| Figure | Description |
|--------|-------------|
| `phase_portrait.png` | (D, h_μ) trajectories across topologies |
| `isoentropic_comparison.png` | Same entropy, different dimension |
| `isodimensional_experiment.png` | Same dimension, different entropy |
| `spectral_theory.png` | Laplacian spectra explaining mechanism |

## Running Simulations

```bash
cd code
python3 spectral_decoupling.py
```

## Files

```
44_spectral_decoupling/
├── paper/
│   ├── spectral_decoupling.tex   # Main manuscript
│   └── references.bib            # Bibliography (includes foundational lit)
├── code/
│   └── spectral_decoupling.py    # Simulations and figures
├── figures/                       # Generated figures
└── README.md
```

## Strategic Notes

This paper serves as a **keystone citation** for other papers in the research program that make dimension-related claims. It provides formal backing for:

- The distinction between "high-dimensional substrate" and "low-dimensional bottleneck"
- Why entropy measures don't capture geometric capacity
- The spectral basis for dimensional constraints

## Related Papers

- 3_intelligence: High-dimensional coherence (should cite this)
- 20_slow_waves_high_D: Neural dimensionality
- 25_lsd_dimensionality: Psychedelics and D_eff
- 21_dimensional_collapse: Minimal embedding dimension (related math)
