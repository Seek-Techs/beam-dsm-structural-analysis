"""Figure 2.1: structural idealisation of the worked (Oyenuga) beam, drawn from beam_dsm.define_beam().

Usage:  python3 figures/make_figure_2_1.py [output_folder]
Writes figure_2_1_beam_idealisation.png (300 dpi), .svg and .pdf (vector, text kept as text).
Every geometric and load value is read from the package, so the figure cannot disagree with the analysed model.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import beam_dsm as dsm

matplotlib.rcParams["svg.fonttype"] = "none"      # keep labels as text in the SVG so that they can be checked
matplotlib.rcParams["pdf.fonttype"] = 42


def build():
    """Draw the figure from beam_dsm.define_beam() and return (fig, ax)."""
    beam = dsm.define_beam()
    x = np.asarray(beam["nodes"])
    udl = np.asarray(beam["element_udl"])
    nodal = beam["point_loads"]
    supports = beam["support_nodes"]
    labels = {n: "ABCDE"[i] for i, n in enumerate(supports)}

    fig, ax = plt.subplots(figsize=(11, 4.6))
    y0 = 0.0
    ax.plot([x[0], x[-1]], [y0, y0], color="black", lw=4, solid_capstyle="butt", zorder=3)

    # distributed loads: a block over each loaded element, arrows pointing down
    top = 1.0
    for e, w in enumerate(udl):
        if w == 0:
            continue
        a, b = x[e], x[e + 1]
        ax.add_patch(Polygon([[a, y0 + 0.12], [b, y0 + 0.12], [b, y0 + top], [a, y0 + top]], closed=True,
                             facecolor="#cfe3f5", edgecolor="#2b6ca3", lw=1, zorder=1))
        n_arrows = max(2, int(round((b - a) / 0.9)))
        for xa in np.linspace(a, b, n_arrows):
            ax.annotate("", xy=(xa, y0 + 0.12), xytext=(xa, y0 + top), arrowprops=dict(arrowstyle="-|>", color="#2b6ca3", lw=0.9))
        narrow = (b - a) < 2.0          # a narrow element gets a smaller label so that it clears the neighbouring arrow
        ax.text((a + b) / 2 - (0.12 if narrow else 0), y0 + top + 0.08, f"{w:g} kN/m", ha="center", va="bottom",
                fontsize=7.5 if narrow else 9, color="#1c4a73")

    # nodal point loads
    for n, P in nodal.items():
        ax.annotate("", xy=(x[n], y0 + 0.06), xytext=(x[n], y0 + 1.75),
                    arrowprops=dict(arrowstyle="-|>", color="#b03030", lw=2.0))
        ax.text(x[n], y0 + 1.82, f"{P:g} kN", ha="center", va="bottom", fontsize=9.5, color="#8a1f1f", fontweight="bold")

    # supports: triangles at the supported nodes (vertical restraint only)
    for n in supports:
        ax.add_patch(Polygon([[x[n], y0 - 0.02], [x[n] - 0.35, y0 - 0.62], [x[n] + 0.35, y0 - 0.62]], closed=True,
                             facecolor="white", edgecolor="black", lw=1.5, zorder=4))
        ax.plot([x[n] - 0.5, x[n] + 0.5], [y0 - 0.62, y0 - 0.62], color="black", lw=1.5)
        ax.text(x[n], y0 - 0.95, labels[n], ha="center", va="top", fontsize=12, fontweight="bold")

    # nodes: dots with node number below the beam (and coordinate)
    for n, xn in enumerate(x):
        ax.plot(xn, y0, "o", color="white", mec="black", mew=1.5, ms=7, zorder=5)
        shift = {4: -0.35, 5: 0.35}.get(n, 0.0)   # nodes 4 and 5 are 1.6 m apart: shift their labels apart
        ax.text(xn + shift, y0 - 1.38, f"node {n}\nx = {xn:g} m", ha="center", va="top", fontsize=8.5, color="#333333")

    # element numbers (1 to 7) between nodes, above the beam line inside the dimension row
    for e in range(len(x) - 1):
        ax.text((x[e] + x[e + 1]) / 2, y0 + 0.26 if udl[e] == 0 else y0 - 0.22, "", fontsize=1)

    # element labels on a row below the node labels
    ymid = y0 - 2.4
    for e in range(len(x) - 1):
        a, b = x[e], x[e + 1]
        ax.annotate("", xy=(a, ymid), xytext=(b, ymid), arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.8))
        ax.text((a + b) / 2, ymid - 0.1, f"element {e + 1}\n{b - a:g} m", ha="center", va="top", fontsize=8, color="#555555")

    # support spans on a row above
    yspan = y0 + 2.75
    sx = [x[n] for n in supports]
    for i in range(len(sx) - 1):
        ax.annotate("", xy=(sx[i], yspan), xytext=(sx[i + 1], yspan), arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
        ax.text((sx[i] + sx[i + 1]) / 2, yspan + 0.08, f"span {labels[supports[i]]}{labels[supports[i + 1]]} = {sx[i + 1] - sx[i]:g} m",
                ha="center", va="bottom", fontsize=9.5)

    ax.set_xlim(x[0] - 1.2, x[-1] + 1.2)
    ax.set_ylim(ymid - 1.2, yspan + 0.8)
    ax.axis("off")
    ax.text(x[0] - 1.1, ymid - 1.15, "Loads act downward.  Supports restrain vertical displacement only; rotation is free.  "
            "Total applied load = 1050 kN.  EI is constant.", fontsize=8.5, ha="left", va="top", color="#333333")

    total = float(np.sum(udl * np.diff(x)) + sum(nodal.values()))
    assert abs(total - 1050.0) < 1e-9, total
    assert len(supports) == 5 and len(x) == 8
    return fig, ax


def main(folder="."):
    os.makedirs(folder, exist_ok=True)
    fig, ax = build()
    base = os.path.join(folder, "figure_2_1_beam_idealisation")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".svg", bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    print("saved", base + ".png/.svg/.pdf")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figures/output")
