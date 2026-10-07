"""
Figures for the SSH model: bulk bands, finite-chain spectrum, edge state.
Usage:  python plot_ssh.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ssh_model import bulk_bands, finite_spectrum, winding_number

NAVY, GOLD, EMER = "#1c4571", "#b0822f", "#1f6b53"


def main():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))

    # (1) bulk bands, trivial vs topological
    for (v, w), c, lab in [((1.0, 0.4), GOLD, "trivial  v>w"),
                           ((0.4, 1.0), NAVY, "topological  w>v")]:
        k, bands = bulk_bands(v, w)
        ax[0].plot(k, bands[0], color=c, lw=1.6)
        ax[0].plot(k, bands[1], color=c, lw=1.6,
                   label=f"{lab}  (ν={winding_number(v, w)})")
    ax[0].set_title("Bulk band structure", color=NAVY, fontweight="bold")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("Energy  E/t")
    ax[0].axhline(0, color="gray", lw=.6); ax[0].legend(fontsize=8); ax[0].grid(alpha=.2)

    # (2) finite-chain spectrum vs dimerisation
    ratios = np.linspace(0.0, 2.0, 60)   # w/v
    v = 1.0
    allE = []
    for r in ratios:
        E, _ = finite_spectrum(v, v * r, n_cells=20)
        allE.append(E)
    allE = np.array(allE)
    for col in range(allE.shape[1]):
        ax[1].plot(ratios, allE[:, col], color=NAVY, lw=.5, alpha=.6)
    ax[1].axvline(1.0, color=GOLD, ls="--", lw=1, label="transition w=v")
    ax[1].set_title("Finite chain: edge states appear for w>v", color=NAVY, fontweight="bold")
    ax[1].set_xlabel("dimerisation  w / v"); ax[1].set_ylabel("Energy  E/t")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.2)

    # (3) the two zero-energy edge-state profiles in the topological phase
    E, V = finite_spectrum(0.4, 1.0, n_cells=20)
    zero_idx = np.argsort(np.abs(E))[:2]
    for i, c in zip(zero_idx, [EMER, GOLD]):
        ax[2].plot(np.abs(V[:, i])**2, color=c, lw=1.4,
                   label=f"E = {E[i]:+.1e}")
    ax[2].set_title("Protected edge states  |ψ|²", color=NAVY, fontweight="bold")
    ax[2].set_xlabel("site index"); ax[2].set_ylabel("probability")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.2)

    fig.tight_layout()
    fig.savefig("figures/ssh_overview.png", dpi=130)
    print("saved figures/ssh_overview.png")


if __name__ == "__main__":
    main()
