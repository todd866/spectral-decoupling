# Spectral Decoupling of Capacity and Entropy in Network Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Status:** Tutorial paper draft complete
**Target Journal:** Journal of Complex Networks (OUP)

## Overview

A **tutorial synthesis** making the classical dimension-entropy distinction explicit for network dynamics via Laplacian spectral analysis.

**Core insight (classical):** Dimension and entropy are independent characteristics of dynamical systems (Eckmann & Ruelle, 1985).

**Network-native formulation:**
- **Spectral capacity** $C(\lambda^*)$: Count of non-trivial Laplacian modes with $\lambda_k < \lambda^*$. Determined purely by **topology**.
- **State entropy proxy**: Spread of stationary distribution = $2\log\sigma + B(G)$, where noise contributes an **additive shift** and topology sets the **baseline offset**.

**Key result:** Networks can have matched state entropy with different capacity (and vice versa). This is proven analytically using the closed-form entropy decomposition.

## Key Equations

**Spectral capacity:**
$$C(\lambda^*) = |\{k : 0 < \lambda_k < \lambda^*\}|$$

**Entropy decomposition:**
$$H = 2\log\sigma - \frac{1}{n-1}\sum_{k=2}^{n} \log(\alpha\lambda_k(2 - \alpha\lambda_k))$$

## Results Summary

At threshold $\lambda^* = 0.1$ for $n=100$ node networks (30 realizations per topology):

| Topology | Capacity $C$ | Why |
|----------|--------------|-----|
| Ring | 6 | Dense small-λ spectrum |
| Modular | 3 | Community modes near zero |
| Small-world | 2 | Rewiring increases spectral gap |
| Random/Scale-free | 0 | Large spectral gap |

## Figures

| Figure | Description |
|--------|-------------|
| `phase_portrait.png` | $(C, H)$ phase space with error bars and σ annotations |
| `isoentropic_comparison.png` | Same entropy, different capacity |
| `isodimensional_experiment.png` | Fixed capacity, varying entropy |
| `spectral_theory.png` | Integrated density of states (eigenvalue CDF) |
| `capacity_curve.png` | Capacity as function of threshold λ* |

## Running Simulations

```bash
cd code
pip install -r requirements.txt
python3 spectral_decoupling.py
```

**Requirements:** See `code/requirements.txt` (numpy, scipy, networkx, matplotlib)

## Paper

- **Title:** Spectral Decoupling of Capacity and Entropy in Network Dynamics
- **PDF:** [spectral_decoupling.pdf](paper/spectral_decoupling.pdf)
- **Pages:** 9 (including 5 figures)

### Technical improvements (v2)
- Stationarity fix: dynamics restricted to nontrivial modes (projecting out λ₁=0 random walk)
- Closed-form entropy decomposition with rigorous proof
- Error bars over 30 graph realizations per topology
- Capacity curve showing threshold sensitivity
- CDF representation for spectral density

## Related Work

This tutorial provides foundational backing for capacity/dimension claims in:

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
  note = {Tutorial paper, in preparation}
}
```

## License

MIT
