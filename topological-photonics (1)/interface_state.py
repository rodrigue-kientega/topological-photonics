"""
Topological interface state in a 1D photonic crystal (FDFD)
===========================================================

We build a dielectric profile epsilon(x) made of N cells of crystal A joined to
N cells of crystal B (the two have swapped layer order, so their shared gap
carries different Zak phases). Solving the 1D Helmholtz eigenproblem

        -d^2E/dx^2 = (omega/c)^2 epsilon(x) E

by finite differences, a mode appears INSIDE the common band gap, strongly
localised at the A|B interface — the photonic analogue of the SSH edge state
and the operating principle of a topological waveguide.

Author: Wendlasida Rodrigue Kientega
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

NAVY, GOLD, EMER = "#1c4571", "#b0822f", "#1f6b53"


def build_epsilon(n_cells=8, dx=0.01):
    """epsilon(x): N cells of A (hi|lo) then N cells of B (lo|hi)."""
    eps_hi, eps_lo = 9.0, 2.25        # n = 3.0 and 1.5
    d_hi, d_lo = 0.5, 1.0
    seg = []
    # crystal A: high-index layer then low-index layer
    for _ in range(n_cells):
        seg += [(eps_hi, d_hi), (eps_lo, d_lo)]
    # crystal B: swapped order
    for _ in range(n_cells):
        seg += [(eps_lo, d_lo), (eps_hi, d_hi)]
    xs, eps = [], []
    x = 0.0
    for e, d in seg:
        npts = int(round(d / dx))
        xs.append(np.full(npts, e))
        x += d
    eps = np.concatenate(xs)
    x_axis = np.arange(eps.size) * dx
    return x_axis, eps


def solve_modes_near(eps, dx, omega_target, n_modes=12):
    """
    FD eigenproblem  (1/eps)(-D2) E = omega^2 E, solved with shift-invert
    around omega_target^2 so we get modes INSIDE the chosen band gap rather
    than the long-wavelength fundamental modes.
    """
    N = eps.size
    main = 2.0 * np.ones(N) / dx**2
    off = -1.0 * np.ones(N - 1) / dx**2
    D2 = diags([off, main, off], [-1, 0, 1])
    Minv = diags(1.0 / eps)
    op = (Minv @ D2).tocsc()
    w, V = eigsh(op, k=n_modes, sigma=omega_target**2, which="LM")
    omega = np.sqrt(np.clip(w, 0, None))
    order = np.argsort(omega)
    return omega[order], V[:, order]


def first_gap_center(dx):
    """Mid-gap frequency of crystal A's lowest band gap (from transfer matrix)."""
    from photonic_crystal_zak import band_structure
    oms = np.linspace(0.05, 4.0, 1200)
    _, gapA = band_structure(3.0, 0.5, 1.5, 1.0, oms)
    # first contiguous gap run
    idx = np.where(gapA)[0]
    if idx.size == 0:
        return 1.0
    start = idx[0]
    end = start
    for k in idx:
        if k == end or k == end + 1:
            end = k
        else:
            break
    return 0.5 * (oms[start] + oms[end])


def main():
    dx = 0.01
    x, eps = build_epsilon(n_cells=8, dx=dx)
    omega_g = first_gap_center(dx)
    print(f"Targeting the first band gap, centre omega ~ {omega_g:.3f}")
    omega, V = solve_modes_near(eps, dx, omega_g, n_modes=16)

    interface_x = x[eps.size // 2]
    # localisation score: fraction of |E|^2 within +/-2 units of the interface
    mask = np.abs(x - interface_x) < 2.0
    loc = [np.sum(np.abs(V[:, i])**2 * mask) / np.sum(np.abs(V[:, i])**2)
           for i in range(V.shape[1])]
    j = int(np.argmax(loc))
    print(f"Most localised mode: index {j}, omega = {omega[j]:.3f}, "
          f"localisation = {loc[j]*100:.0f}% within ±2 of interface")

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[1, 2.2], sharex=True)
    ax[0].fill_between(x, eps, step="mid", color="#dbe3ee")
    ax[0].axvline(interface_x, color=GOLD, lw=1.5, ls="--")
    ax[0].set_ylabel("ε(x)"); ax[0].set_title("Dielectric profile:  crystal A | crystal B",
                                              color=NAVY, fontweight="bold")

    E = V[:, j]; E = E / np.max(np.abs(E))
    ax[1].plot(x, np.abs(E)**2, color=EMER, lw=1.5)
    ax[1].axvline(interface_x, color=GOLD, lw=1.5, ls="--", label="A|B interface")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("|E(x)|²")
    ax[1].set_title(f"Topological interface state  (ω = {omega[j]:.3f}), "
                    f"localised at the junction", color=NAVY, fontweight="bold")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig("figures/interface_state.png", dpi=130)
    print("saved figures/interface_state.png")


if __name__ == "__main__":
    main()
