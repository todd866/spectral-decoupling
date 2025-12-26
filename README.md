# Spectral Decoupling of Capacity and Entropy in Network Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Status:** Draft complete
**Target Journal:** Journal of Complex Networks (OUP)

## Overview

Spectral capacity and state entropy are **independent axes** of network dynamical complexity:

- **Spectral capacity** $C(\lambda^*)$: Count of non-trivial Laplacian modes with $\lambda_k < \lambda^*$. Determined purely by **topology**.
- **State entropy**: Spread of stationary distribution. Determined by **noise**.

**Key result:** Networks can have matched state entropy with different capacity (and vice versa). The Laplacian eigenvalue density controls this decoupling.

## Key Equation

$$C(\lambda^*) = |\{k : 0 < \lambda_k < \lambda^*\}|$$

This is a **capacity measure** (slow-mode count), NOT a fractal dimension.

## Results Summary

At threshold $\lambda^* = 0.1$ for $n=100$ node networks:

| Topology | Capacity $C$ | Why |
|----------|--------------|-----|
| Ring | 6 | Dense small-λ spectrum |
| Modular | 3 | Community modes near zero |
| Small-world | 2 | Rewiring increases spectral gap |
| Random/Scale-free | 0 | Large spectral gap |

## Figures

| Figure | Description |
|--------|-------------|
| `phase_portrait.png` | Each topology at fixed capacity; noise moves entropy vertically |
| `isoentropic_comparison.png` | Same entropy, different capacity |
| `isodimensional_experiment.png` | Fixed capacity, varying entropy |
| `spectral_theory.png` | Laplacian spectra with threshold |

## Running Simulations

```bash
cd code
python3 spectral_decoupling.py
```

**Requirements:** numpy, scipy, networkx, matplotlib

## Paper

- **Title:** Spectral Decoupling of Capacity and Entropy in Network Dynamics
- **PDF:** [spectral_decoupling.pdf](paper/spectral_decoupling.pdf)

## Related Work

This paper provides foundational backing for capacity/dimension claims in:

| Paper | Repository |
|-------|------------|
| Minimal Embedding Dimension | [todd866/minimalembeddingdimension](https://github.com/todd866/minimalembeddingdimension) |
| LSD/Psychedelics Dimensionality | [todd866/lsd-dimensionality](https://github.com/todd866/lsd-dimensionality) |

## Citation

```bibtex
@article{todd2025spectral,
  author = {Todd, Ian},
  title = {Spectral Decoupling of Capacity and Entropy in Network Dynamics},
  journal = {Journal of Complex Networks},
  year = {2025},
  note = {In preparation}
}
```

## License

MIT
