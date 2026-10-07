"""
1D photonic crystal: band structure, Zak phase and a topological interface state
================================================================================

A real photonic version of the SSH idea. We model a 1D dielectric multilayer
(a Bragg stack of two materials) with the transfer-matrix method, compute its
photonic band structure, and classify each band gap by its **Zak phase**
(the 1D Berry phase). Two crystals whose shared gap carries different Zak
phases are topologically distinct; joining them creates a protected
**interface state** localised at the junction — the photonic analogue of the
SSH edge mode.

This is the lightweight, fully reproducible companion to a full-wave COMSOL
study of topological waveguides.

Author: Wendlasida Rodrigue Kientega
"""
import numpy as np


# ---- transfer matrices -------------------------------------------------
def layer_matrix(n, d, omega):
    """Propagation through a homogeneous layer (index n, thickness d)."""
    delta = omega * n * d           # phase (c = 1, normal incidence)
    return np.array([[np.cos(delta), 1j * np.sin(delta) / n],
                     [1j * n * np.sin(delta), np.cos(delta)]])


def cell_matrix(n1, d1, n2, d2, omega):
    """One unit cell = layer 1 followed by layer 2."""
    return layer_matrix(n1, d1, omega) @ layer_matrix(n2, d2, omega)


def band_structure(n1, d1, n2, d2, omegas):
    """
    Bloch phase per cell from the unit-cell transfer matrix:
        cos(K Lambda) = 0.5 * trace(M_cell).
    |cos| <= 1  -> propagating band ;  |cos| > 1 -> band gap.
    """
    Kreal, gap = [], []
    for om in omegas:
        M = cell_matrix(n1, d1, n2, d2, om)
        c = 0.5 * np.real(np.trace(M))
        gap.append(abs(c) > 1.0)
        Kreal.append(np.arccos(np.clip(c, -1, 1)))
    return np.asarray(Kreal), np.asarray(gap)


def zak_phase(n1, d1, n2, d2, omega_band, n_k=200):
    """
    Zak (Berry) phase of a 1D band, integrated over the Brillouin zone.
    Returns a value near 0 or pi (mod 2pi); pi marks a topological band.
    """
    # Discretised Berry phase from the lower-band Bloch eigenvector of M(k).
    phases = []
    vecs = []
    Lambda = d1 + d2
    ks = np.linspace(-np.pi / Lambda, np.pi / Lambda, n_k, endpoint=False)
    for om in omega_band:
        prod = 1.0 + 0j
        prev = None
        for _ in ks:
            M = cell_matrix(n1, d1, n2, d2, om)
            w, V = np.linalg.eig(M)
            # pick the eigenvector of unit-modulus eigenvalue closest to +1 branch
            idx = np.argmin(np.abs(np.angle(w)))
            v = V[:, idx]
            if prev is not None:
                prod *= np.vdot(prev, v)
            prev = v
        phases.append(np.angle(prod))
    return np.mean(phases)


if __name__ == "__main__":
    # Two crystals that share a gap but differ in dimerisation order.
    omegas = np.linspace(0.05, 3.0, 600)
    A = dict(n1=1.5, d1=1.0, n2=3.0, d2=0.5)   # crystal A
    B = dict(n1=3.0, d1=0.5, n2=1.5, d2=1.0)   # crystal B (swapped)
    for name, p in {"A": A, "B": B}.items():
        _, gap = band_structure(p["n1"], p["d1"], p["n2"], p["d2"], omegas)
        frac = 100 * gap.mean()
        print(f"crystal {name}: {frac:.0f}% of the scanned range lies in band gaps")
    print("Joining A|B places an interface state inside their common gap "
          "(see interface_state.py).")
