# Spectral Decoupling of Dimension and Information in Network Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Status:** Draft complete
**Target Journal:** Journal of Complex Networks (OUP)

## Overview

Geometric dimensionality and statistical information are **independent axes** of network dynamical complexity:

- **Dimension** $D_{\mathrm{eff}}(\tau)$: Count of Laplacian modes with relaxation time > $\tau$. Determined purely by **topology**.
- **Entropy** $h$: Statistical property of the dynamics. Determined by **noise/coupling**.

**Key result:** Networks can have matched entropy with different dimension (and vice versa). The Laplacian eigenvalue density controls this decoupling.

## Key Equation

$$D_{\mathrm{eff}}(\tau) = |\{k : \lambda_k < 1/(\alpha\tau)\}|$$

where $\lambda_k$ are normalized Laplacian eigenvalues, $\alpha$ is the diffusion rate, and $\tau$ is the observation timescale.

## Figures

| Figure | Description |
|--------|-------------|
| `phase_portrait.png` | Each topology at fixed $D_{\mathrm{eff}}$; noise moves entropy vertically |
| `isoentropic_comparison.png` | Same entropy, different dimension |
| `isodimensional_experiment.png` | Fixed dimension, varying entropy |
| `spectral_theory.png` | Laplacian spectra with threshold |

## Running Simulations

```bash
cd code
python3 spectral_decoupling.py
```

**Requirements:** numpy, scipy, networkx, matplotlib

## Paper

- **Title:** Spectral Decoupling of Dimension and Information in Network Dynamics
- **Status:** Draft complete
- **PDF:** [spectral_decoupling.pdf](paper/spectral_decoupling.pdf)

## Related Papers

This paper provides foundational backing for dimension-related claims in:

| Paper | Repository |
|-------|------------|
| Intelligence as High-Dimensional Coherence | [todd866/high-dimensional-intelligence](https://github.com/todd866/high-dimensional-intelligence) |
| Minimal Embedding Dimension | [todd866/minimalembeddingdimension](https://github.com/todd866/minimalembeddingdimension) |
| LSD/Psychedelics Dimensionality | [todd866/lsd-dimensionality](https://github.com/todd866/lsd-dimensionality) |
| Cortical Oscillations | [todd866/slow-waves-high-d](https://github.com/todd866/slow-waves-high-d) |

## Citation

```bibtex
@article{todd2025spectral,
  author = {Todd, Ian},
  title = {Spectral Decoupling of Dimension and Information in Network Dynamics},
  journal = {Journal of Complex Networks},
  year = {2025},
  note = {In preparation}
}
```

## License

MIT
