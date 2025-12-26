#!/usr/bin/env python3
"""
Spectral Decoupling Analysis: Dimension vs Information on Networks

Key insight: Spectral D_eff (slow-mode count) is a CAPACITY measure determined
by topology. It is NOT a fractal dimension in the Grassberger-Procaccia sense.

Fixed issues:
- λ* threshold now in small-λ regime to separate topologies
- Entropy proxy clearly labeled (not called h_μ)
- Relative paths for portability
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import networkx as nx
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False
})


def create_graph(graph_type: str, n: int = 100) -> nx.Graph:
    """Create connected graph of given topology."""
    if graph_type == 'ring':
        # Ring lattice: dense small-λ spectrum (many slow modes)
        G = nx.watts_strogatz_graph(n, 6, 0.0)
    elif graph_type == 'small_world':
        # Some rewiring: intermediate
        G = nx.watts_strogatz_graph(n, 6, 0.1)
    elif graph_type == 'modular':
        # Strong communities: near-degenerate modes at small λ
        sizes = [n//4] * 4
        p_in, p_out = 0.5, 0.005
        probs = [[p_in if i==j else p_out for j in range(4)] for i in range(4)]
        G = nx.stochastic_block_model(sizes, probs)
    elif graph_type == 'random':
        # Large spectral gap: few slow modes
        G = nx.erdos_renyi_graph(n, 0.15)
    elif graph_type == 'scale_free':
        G = nx.barabasi_albert_graph(n, 3)
    else:
        raise ValueError(f"Unknown: {graph_type}")

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def get_laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    """Normalized Laplacian eigenvalues, sorted."""
    L = nx.normalized_laplacian_matrix(G).toarray()
    return np.sort(linalg.eigvalsh(L))


def spectral_capacity(eigenvalues: np.ndarray, lambda_star: float = 0.1) -> int:
    """
    Spectral capacity: count of non-trivial modes with λ < λ*.

    This counts slow modes (those with relaxation time > 1/(α λ*)).
    We EXCLUDE the trivial λ₁=0 mode (constant eigenvector).

    NOTE: This is a CAPACITY measure, not a fractal dimension.
    """
    # Exclude trivial zero mode (λ₁ ≈ 0)
    nontrivial = eigenvalues[eigenvalues > 1e-8]
    return int(np.sum(nontrivial < lambda_star))


def simulate_and_get_state_entropy(G: nx.Graph, alpha: float, sigma: float, T: int = 8000) -> float:
    """
    Simulate diffusion and return STATE ENTROPY PROXY.

    We compute: (1/n) * log det(Cov) = mean(log(eigenvalues of Cov))

    This is proportional to differential entropy of the stationary distribution,
    NOT the entropy rate h_μ of the process.
    """
    n = G.number_of_nodes()
    L = nx.normalized_laplacian_matrix(G).toarray()
    A = np.eye(n) - alpha * L

    # Ensure stability
    if np.max(np.abs(linalg.eigvalsh(A))) >= 1:
        alpha = 0.8 / np.max(linalg.eigvalsh(L))
        A = np.eye(n) - alpha * L

    x = np.zeros((T, n))
    x[0] = np.random.randn(n) * 0.1
    for t in range(1, T):
        x[t] = A @ x[t-1] + sigma * np.random.randn(n)

    X = x[T//4:]  # Discard burn-in

    cov = np.cov(X.T)
    eigs = linalg.eigvalsh(cov)
    eigs = eigs[eigs > 1e-10]

    # State entropy proxy: mean log eigenvalue ∝ (1/n) log det(Cov)
    return np.mean(np.log(eigs)) if len(eigs) > 0 else 0.0


def get_output_dir() -> str:
    """Get output directory relative to script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


# =============================================================================
# FIGURE 1: Phase Portrait - Capacity vs State Entropy
# =============================================================================
def generate_phase_portrait(save_path: str):
    """
    Show that topology fixes spectral capacity; noise moves state entropy.
    """
    np.random.seed(42)

    # λ* = 0.1 probes the small-eigenvalue regime where topologies differ
    lambda_star = 0.1

    topologies = {
        'Ring (high capacity)': ('ring', '#3498db'),
        'Small-world': ('small_world', '#e74c3c'),
        'Modular': ('modular', '#2ecc71'),
        'Random (low capacity)': ('random', '#9b59b6'),
        'Scale-free': ('scale_free', '#f39c12')
    }

    n = 100
    alpha = 0.1
    sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    fig, ax = plt.subplots(figsize=(9, 6))

    for label, (gtype, color) in topologies.items():
        G = create_graph(gtype, n)
        eigs = get_laplacian_eigenvalues(G)
        capacity = spectral_capacity(eigs, lambda_star)

        entropies = []
        for sigma in sigma_values:
            h = simulate_and_get_state_entropy(G, alpha, sigma)
            entropies.append(h)

        # Add jitter for visibility (capacity is discrete)
        cap_jitter = capacity + np.random.uniform(-0.3, 0.3)

        ax.plot([cap_jitter]*len(entropies), entropies, 'o-', color=color,
                label=f'{label} (C={capacity})', markersize=8, linewidth=2, alpha=0.8)

        # Arrow showing σ increases
        mid = len(entropies) // 2
        ax.annotate('', xy=(cap_jitter, entropies[mid+1]),
                    xytext=(cap_jitter, entropies[mid]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

    ax.set_xlabel(r'Spectral capacity $C(\lambda^*)$ (slow-mode count, $\lambda^*=0.1$)', fontsize=12)
    ax.set_ylabel(r'State entropy proxy (mean log cov eigenvalue)', fontsize=12)
    ax.set_title('Capacity-Entropy Phase Space\nTopology fixes capacity; noise moves state entropy', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.text(0.03, 0.97, r'$\sigma$ increases $\uparrow$', transform=ax.transAxes,
            fontsize=11, va='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# FIGURE 2: Iso-Entropy, Different Capacity
# =============================================================================
def generate_isoentropic(save_path: str):
    """
    Match state entropy by tuning σ, show spectral capacity differs.
    """
    np.random.seed(42)

    n = 100
    alpha = 0.1
    lambda_star = 0.1
    target_entropy = 1.0

    configs = [
        ('Ring', 'ring', '#3498db'),
        ('Modular', 'modular', '#2ecc71'),
        ('Random', 'random', '#9b59b6')
    ]

    results = {}

    for label, gtype, color in configs:
        G = create_graph(gtype, n)
        eigs = get_laplacian_eigenvalues(G)
        capacity = spectral_capacity(eigs, lambda_star)

        # Binary search for σ that gives target entropy
        sigma_lo, sigma_hi = 0.1, 8.0
        for _ in range(25):
            sigma = (sigma_lo + sigma_hi) / 2
            h = simulate_and_get_state_entropy(G, alpha, sigma, T=6000)
            if h < target_entropy:
                sigma_lo = sigma
            else:
                sigma_hi = sigma

        sigma_final = (sigma_lo + sigma_hi) / 2
        h_final = simulate_and_get_state_entropy(G, alpha, sigma_final, T=10000)

        results[label] = {'capacity': capacity, 'entropy': h_final, 'sigma': sigma_final, 'color': color}
        print(f"{label}: C={capacity}, h={h_final:.2f}, σ={sigma_final:.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = list(results.keys())
    capacities = [results[l]['capacity'] for l in labels]
    entropies = [results[l]['entropy'] for l in labels]
    colors = [results[l]['color'] for l in labels]

    # Left: Capacity differs
    bars1 = axes[0].bar(labels, capacities, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel(r'Spectral capacity $C(\lambda^*)$', fontsize=11)
    axes[0].set_title('Spectral Capacity\n(differs despite matched entropy)', fontsize=11)
    for bar, c in zip(bars1, capacities):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(c), ha='center', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, max(capacities) * 1.3)

    # Right: Entropy matched
    bars2 = axes[1].bar(labels, entropies, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('State entropy proxy', fontsize=11)
    axes[1].set_title('State Entropy\n(matched by tuning $\\sigma$)', fontsize=11)
    axes[1].axhline(y=target_entropy, color='black', linestyle='--', linewidth=2,
                    label=f'Target = {target_entropy}')
    for bar, h in zip(bars2, entropies):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f'{h:.2f}', ha='center', fontsize=10)
    axes[1].legend(loc='upper right')

    plt.suptitle('Iso-Entropy Demonstration: Same State Entropy, Different Capacity',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# FIGURE 3: Fixed Capacity, Variable Entropy
# =============================================================================
def generate_fixed_capacity(save_path: str):
    """
    Fix topology (fixes capacity), vary noise, show state entropy changes.
    """
    np.random.seed(42)

    G = create_graph('modular', n=100)
    eigs = get_laplacian_eigenvalues(G)
    alpha = 0.1
    lambda_star = 0.1
    capacity = spectral_capacity(eigs, lambda_star)

    sigma_values = np.linspace(0.3, 3.5, 15)
    entropies = []

    for sigma in sigma_values:
        h = simulate_and_get_state_entropy(G, alpha, sigma, T=8000)
        entropies.append(h)

    entropies = np.array(entropies)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Capacity is CONSTANT (by definition)
    axes[0].axhline(capacity, color='#2ecc71', linewidth=4)
    axes[0].fill_between(sigma_values, capacity-0.5, capacity+0.5, alpha=0.3, color='#2ecc71')
    axes[0].set_xlabel(r'Noise level $\sigma$', fontsize=11)
    axes[0].set_ylabel(r'Spectral capacity $C(\lambda^*)$', fontsize=11)
    axes[0].set_title('Spectral Capacity\n(constant: determined by topology alone)', fontsize=11)
    axes[0].set_ylim(0, capacity * 2)
    axes[0].text(0.5, 0.85, f'$C = {capacity}$\n(invariant to noise)',
                 transform=axes[0].transAxes, fontsize=12, ha='center',
                 bbox=dict(facecolor='#2ecc71', alpha=0.3))

    # Right: State entropy varies with σ
    axes[1].plot(sigma_values, entropies, 's-', color='#e74c3c', markersize=7, linewidth=2)
    axes[1].set_xlabel(r'Noise level $\sigma$', fontsize=11)
    axes[1].set_ylabel('State entropy proxy', fontsize=11)
    axes[1].set_title('State Entropy\n(varies continuously with noise)', fontsize=11)

    h_range = np.max(entropies) - np.min(entropies)
    axes[1].text(0.05, 0.95, f'Range = {h_range:.2f}', transform=axes[1].transAxes,
                 fontsize=11, va='top', bbox=dict(facecolor='#e74c3c', alpha=0.3))

    plt.suptitle('Fixed Capacity, Variable Entropy (Modular Graph)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# FIGURE 4: Spectral Theory - Eigenvalue Distributions
# =============================================================================
def generate_spectral_theory(save_path: str):
    """Laplacian eigenvalue distributions showing capacity threshold."""
    np.random.seed(42)

    n = 100
    lambda_star = 0.1  # Threshold in small-λ regime

    configs = [
        ('Ring (high C)', 'ring', '#3498db'),
        ('Modular', 'modular', '#2ecc71'),
        ('Random (low C)', 'random', '#9b59b6'),
        ('Scale-free', 'scale_free', '#f39c12')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, (label, gtype, color) in zip(axes.flat, configs):
        G = create_graph(gtype, n)
        eigs = get_laplacian_eigenvalues(G)
        capacity = spectral_capacity(eigs, lambda_star)

        # Focus on small-λ region for clarity
        ax.hist(eigs, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(lambda_star, color='#c0392b', linestyle='--', linewidth=2.5,
                   label=f'$\\lambda^* = {lambda_star}$')

        # Shade region below threshold
        ax.axvspan(0, lambda_star, alpha=0.2, color='#c0392b')

        ax.text(0.95, 0.95, f'$C(\\lambda^*) = {capacity}$',
                transform=ax.transAxes, fontsize=12, va='top', ha='right',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

        ax.set_xlabel(r'Laplacian eigenvalue $\lambda$', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{label} (n={len(eigs)})', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(-0.05, 2.1)

    plt.suptitle(r'Spectral Capacity: $C(\lambda^*) = |\{k : 0 < \lambda_k < \lambda^*\}|$' +
                 '\nModes below threshold (shaded) are slow modes',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Generating figures with corrected spectral capacity...")
    print("λ* = 0.1 (probes small-eigenvalue regime)")
    print("=" * 60)

    fig_dir = get_output_dir()

    print("\n[1/4] Phase portrait...")
    generate_phase_portrait(os.path.join(fig_dir, 'phase_portrait.png'))

    print("\n[2/4] Iso-entropy...")
    generate_isoentropic(os.path.join(fig_dir, 'isoentropic_comparison.png'))

    print("\n[3/4] Fixed capacity...")
    generate_fixed_capacity(os.path.join(fig_dir, 'isodimensional_experiment.png'))

    print("\n[4/4] Spectral theory...")
    generate_spectral_theory(os.path.join(fig_dir, 'spectral_theory.png'))

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
