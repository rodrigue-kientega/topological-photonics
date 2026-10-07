"""
Su–Schrieffer–Heeger (SSH) model — the simplest topological system
==================================================================

The SSH model is a 1D chain with two alternating couplings (intra-cell v and
inter-cell w). It is the textbook starting point for topological photonics:
the same mathematics describes a dimerised chain of coupled optical waveguides
or resonators.

This module computes
  * the bulk band structure E(k) = ± |v + w e^{ik}|,
  * the winding number (topological invariant) of the off-diagonal term,
  * the spectrum of a finite chain, which hosts protected zero-energy EDGE
    states when the chain is in the topological phase (w > v).

Author: Wendlasida Rodrigue Kientega
"""
import numpy as np


def bulk_bands(v, w, n_k=400):
    """Two-band dispersion of the infinite SSH chain."""
    k = np.linspace(-np.pi, np.pi, n_k)
    q = v + w * np.exp(1j * k)          # off-diagonal Bloch term
    E = np.abs(q)
    return k, np.vstack([-E, E])


def winding_number(v, w, n_k=2000):
    """
    Topological invariant: winding of d(k) = (Re q, Im q) around the origin.
    Returns 0 (trivial, v > w) or 1 (topological, w > v).
    """
    k = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    q = v + w * np.exp(1j * k)
    phase = np.unwrap(np.angle(q))
    return int(round((phase[-1] - phase[0]) / (2 * np.pi)))


def finite_chain_hamiltonian(v, w, n_cells):
    """Real-space Hamiltonian of a finite open SSH chain (2*n_cells sites)."""
    N = 2 * n_cells
    H = np.zeros((N, N))
    for c in range(n_cells):
        a = 2 * c
        H[a, a + 1] = H[a + 1, a] = v           # intra-cell hop
        if c < n_cells - 1:
            H[a + 1, a + 2] = H[a + 2, a + 1] = w  # inter-cell hop
    return H


def finite_spectrum(v, w, n_cells=20):
    """Eigenvalues and eigenvectors of the finite chain (sorted by energy)."""
    E, V = np.linalg.eigh(finite_chain_hamiltonian(v, w, n_cells))
    return E, V


if __name__ == "__main__":
    for label, (v, w) in {"trivial (v>w)": (1.0, 0.4),
                          "topological (w>v)": (0.4, 1.0)}.items():
        nu = winding_number(v, w)
        E, _ = finite_spectrum(v, w, n_cells=20)
        n_zero = int(np.sum(np.abs(E) < 1e-6))
        print(f"{label:22s}  winding = {nu}   near-zero edge modes = {n_zero}")
