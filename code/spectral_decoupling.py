#!/usr/bin/env python3
"""
Spectral Decoupling Analysis: Dimension vs Information on Networks

Key insight: Use SPECTRAL D_eff (Laplacian mode count) as the geometric dimension,
not participation ratio from covariance. Spectral D_eff is determined purely by
topology; entropy is determined by dynamics/noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import networkx as nx
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
        G = nx.watts_strogatz_graph(n, 4, 0.0)
    elif graph_type == 'small_world':
        G = nx.watts_strogatz_graph(n, 6, 0.3)
    elif graph_type == 'modular':
        sizes = [n//4] * 4
        p_in, p_out = 0.5, 0.01
        probs = [[p_in if i==j else p_out for j in range(4)] for i in range(4)]
        G = nx.stochastic_block_model(sizes, probs)
    elif graph_type == 'random':
        G = nx.erdos_renyi_graph(n, 0.1)
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


def spectral_deff(eigenvalues: np.ndarray, alpha: float = 0.1, tau: float = 10.0) -> int:
    """
    Spectral effective dimension: count of modes with relaxation time > tau.
    D_eff(τ) = |{k : λ_k < 1/(ατ)}|

    This is PURELY topological - doesn't depend on noise or dynamics.
    """
    threshold = 1.0 / (alpha * tau)
    return int(np.sum(eigenvalues < threshold))


def simulate_and_get_entropy(G: nx.Graph, alpha: float, sigma: float, T: int = 8000) -> float:
    """
    Simulate diffusion dynamics and return entropy proxy.
    Entropy = mean log eigenvalue of covariance (scales with differential entropy).
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

    # Discard burn-in
    X = x[T//4:]

    # Entropy proxy from covariance
    cov = np.cov(X.T)
    eigs = linalg.eigvalsh(cov)
    eigs = eigs[eigs > 1e-10]
    return np.mean(np.log(eigs)) if len(eigs) > 0 else 0.0


# =============================================================================
# FIGURE 1: Phase Portrait
# =============================================================================
def generate_phase_portrait(save_path: str):
    """
    Show that different topologies occupy different D_eff values,
    while noise moves them along the entropy axis.
    """
    np.random.seed(42)

    topologies = {
        'Modular (high D)': ('modular', '#2ecc71'),
        'Ring': ('ring', '#3498db'),
        'Small-world': ('small_world', '#e74c3c'),
        'Random (ER)': ('random', '#9b59b6'),
        'Scale-free': ('scale_free', '#f39c12')
    }

    n = 100
    alpha = 0.1
    tau = 10.0
    sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    fig, ax = plt.subplots(figsize=(9, 6))

    for label, (gtype, color) in topologies.items():
        G = create_graph(gtype, n)
        eigs = get_laplacian_eigenvalues(G)
        d_spec = spectral_deff(eigs, alpha, tau)

        # Get entropy at different noise levels
        entropies = []
        for sigma in sigma_values:
            h = simulate_and_get_entropy(G, alpha, sigma)
            entropies.append(h)

        # Plot: D_eff is FIXED by topology, only h varies
        # Add small jitter to D for visibility
        d_jitter = d_spec + np.random.uniform(-0.3, 0.3)

        ax.plot([d_jitter]*len(entropies), entropies, 'o-', color=color,
                label=f'{label} (D={d_spec})', markersize=8, linewidth=2, alpha=0.8)

        # Arrow showing direction
        mid = len(entropies) // 2
        ax.annotate('', xy=(d_jitter, entropies[mid+1]),
                    xytext=(d_jitter, entropies[mid]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

    ax.set_xlabel(r'Spectral dimension $D_{\mathrm{eff}}(\tau)$ (mode count)', fontsize=12)
    ax.set_ylabel(r'Entropy proxy $h$ (mean log eigenvalue)', fontsize=12)
    ax.set_title('Dimension-Entropy Phase Space\nTopology fixes D; noise moves h', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.text(0.03, 0.97, r'$\sigma$ increases $\uparrow$', transform=ax.transAxes,
            fontsize=11, va='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# FIGURE 2: Iso-Entropy, Different D
# =============================================================================
def generate_isoentropic(save_path: str):
    """
    Match entropy by tuning sigma, show spectral D_eff differs.
    """
    np.random.seed(42)

    n = 100
    alpha = 0.1
    tau = 10.0
    target_h = 0.5

    configs = [
        ('Modular', 'modular', '#2ecc71'),
        ('Ring', 'ring', '#3498db'),
        ('Random', 'random', '#9b59b6')
    ]

    results = {}

    for label, gtype, color in configs:
        G = create_graph(gtype, n)
        eigs = get_laplacian_eigenvalues(G)
        d_spec = spectral_deff(eigs, alpha, tau)

        # Binary search for sigma that gives target entropy
        sigma_lo, sigma_hi = 0.1, 5.0
        for _ in range(20):
            sigma = (sigma_lo + sigma_hi) / 2
            h = simulate_and_get_entropy(G, alpha, sigma, T=6000)
            if h < target_h:
                sigma_lo = sigma
            else:
                sigma_hi = sigma

        sigma_final = (sigma_lo + sigma_hi) / 2
        h_final = simulate_and_get_entropy(G, alpha, sigma_final, T=8000)

        results[label] = {'D_spec': d_spec, 'h': h_final, 'sigma': sigma_final, 'color': color}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = list(results.keys())
    d_specs = [results[l]['D_spec'] for l in labels]
    h_vals = [results[l]['h'] for l in labels]
    colors = [results[l]['color'] for l in labels]

    # Left: D_eff differs (this is the point!)
    bars1 = axes[0].bar(labels, d_specs, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel(r'Spectral $D_{\mathrm{eff}}(\tau)$ (mode count)', fontsize=11)
    axes[0].set_title('Geometric Dimension\n(differs despite matched entropy)', fontsize=11)
    for bar, d in zip(bars1, d_specs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(d), ha='center', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, max(d_specs) * 1.3)

    # Right: Entropy matched
    bars2 = axes[1].bar(labels, h_vals, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel(r'Entropy proxy $h$', fontsize=11)
    axes[1].set_title('Entropy\n(matched by tuning $\\sigma$)', fontsize=11)
    axes[1].axhline(y=target_h, color='black', linestyle='--', linewidth=2,
                    label=f'Target = {target_h}')
    for bar, h in zip(bars2, h_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f'{h:.2f}', ha='center', fontsize=10)
    axes[1].legend(loc='upper right')

    plt.suptitle('Iso-Entropy Demonstration: Same $h$, Different $D_{\\mathrm{eff}}$',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# FIGURE 3: Iso-Dimension, Different Entropy
# =============================================================================
def generate_isodimensional(save_path: str):
    """
    Fix topology (fixes D_eff), vary noise, show h changes.
    Spectral D_eff is CONSTANT by definition for fixed topology.
    """
    np.random.seed(42)

    G = create_graph('modular', n=100)
    eigs = get_laplacian_eigenvalues(G)
    alpha = 0.1
    tau = 10.0
    d_spec = spectral_deff(eigs, alpha, tau)

    sigma_values = np.linspace(0.3, 3.0, 15)
    h_vals = []

    for sigma in sigma_values:
        h = simulate_and_get_entropy(G, alpha, sigma, T=8000)
        h_vals.append(h)

    h_vals = np.array(h_vals)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: D_eff is CONSTANT (by definition - it's spectral)
    axes[0].axhline(d_spec, color='#2ecc71', linewidth=3)
    axes[0].fill_between(sigma_values, d_spec-0.5, d_spec+0.5, alpha=0.3, color='#2ecc71')
    axes[0].set_xlabel(r'Noise level $\sigma$', fontsize=11)
    axes[0].set_ylabel(r'Spectral $D_{\mathrm{eff}}(\tau)$', fontsize=11)
    axes[0].set_title('Geometric Dimension\n(fixed by topology)', fontsize=11)
    axes[0].set_ylim(0, d_spec * 1.5)
    axes[0].text(0.5, 0.85, f'$D_{{\\mathrm{{eff}}}} = {d_spec}$\n(constant)',
                 transform=axes[0].transAxes, fontsize=12, ha='center',
                 bbox=dict(facecolor='#2ecc71', alpha=0.3))

    # Right: h varies with sigma
    axes[1].plot(sigma_values, h_vals, 's-', color='#e74c3c', markersize=7, linewidth=2)
    axes[1].set_xlabel(r'Noise level $\sigma$', fontsize=11)
    axes[1].set_ylabel(r'Entropy proxy $h$', fontsize=11)
    axes[1].set_title('Entropy\n(varies with noise)', fontsize=11)

    h_range = np.max(h_vals) - np.min(h_vals)
    axes[1].text(0.05, 0.95, f'Range = {h_range:.2f}', transform=axes[1].transAxes,
                 fontsize=11, va='top', bbox=dict(facecolor='#e74c3c', alpha=0.3))

    plt.suptitle('Iso-Dimension Demonstration: Fixed $D_{\\mathrm{eff}}$, Variable $h$\n(Modular topology)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# FIGURE 4: Spectral Theory
# =============================================================================
def generate_spectral_theory(save_path: str):
    """Laplacian eigenvalue distributions with D_eff threshold."""
    np.random.seed(42)

    n = 100
    alpha = 0.1
    tau = 10.0
    threshold = 1.0 / (alpha * tau)

    configs = [
        ('Modular', 'modular', '#2ecc71'),
        ('Ring', 'ring', '#3498db'),
        ('Random (ER)', 'random', '#9b59b6'),
        ('Scale-free', 'scale_free', '#f39c12')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, (label, gtype, color) in zip(axes.flat, configs):
        G = create_graph(gtype, n)
        eigs = get_laplacian_eigenvalues(G)
        d_spec = spectral_deff(eigs, alpha, tau)

        ax.hist(eigs, bins=25, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(threshold, color='#c0392b', linestyle='--', linewidth=2.5,
                   label=f'$\\lambda^* = {threshold:.1f}$')

        # Shade region below threshold
        ax.axvspan(0, threshold, alpha=0.15, color='#c0392b')

        ax.text(0.95, 0.95, f'$D_{{\\mathrm{{eff}}}}(\\tau) = {d_spec}$',
                transform=ax.transAxes, fontsize=12, va='top', ha='right',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

        ax.set_xlabel(r'Laplacian eigenvalue $\lambda$', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{label} (n={len(eigs)})', fontsize=11)
        ax.legend(loc='upper left', fontsize=9)

    plt.suptitle(r'Spectral Dimension: $D_{\mathrm{eff}}(\tau) = |\{k : \lambda_k < 1/(\alpha\tau)\}|$' +
                 '\nModes below threshold (shaded) contribute to geometric capacity',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Generating figures with SPECTRAL D_eff...")
    print("=" * 60)

    fig_dir = '/Users/iantodd/Desktop/highdimensional/44_spectral_decoupling/figures'

    print("\n[1/4] Phase portrait...")
    generate_phase_portrait(f'{fig_dir}/phase_portrait.png')

    print("\n[2/4] Iso-entropy...")
    generate_isoentropic(f'{fig_dir}/isoentropic_comparison.png')

    print("\n[3/4] Iso-dimension...")
    generate_isodimensional(f'{fig_dir}/isodimensional_experiment.png')

    print("\n[4/4] Spectral theory...")
    generate_spectral_theory(f'{fig_dir}/spectral_theory.png')

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
