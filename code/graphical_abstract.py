"""
graphical_abstract.py
=====================
Render the Elsevier graphical abstract for
"Resistance-aware precision dosing as posterior inference".

Elsevier spec: a single concise visual, minimum 1328 x 531 px (w x h), readable
at 5 x 13 cm / 96 dpi, sans-serif. We render at 1992 x 797 px (figsize 13.28 x
5.31 in at 150 dpi) -> same 2.5:1 aspect, within the 2000 x 1000 maximum, crisp.

The figure narrates the paper left-to-right in three stages:
  1. genotype -> (MIC, fitness) map (additive + epistasis, interval-censored);
  2. closed-form MPC posterior and the PK coupling T_MSW = t_half x W;
  3. external validation on hollow-fibre data (peak>MPC necessary, not sufficient).

Headline numbers are the deployed results reported in the manuscript
(Table 3 / Section 6.6); the MPC posterior curve is the Gaussian on log2 MPC
implied by E[MPC]=1.17 mg/L and its 95% CI 0.36-3.78 mg/L.

Run: python graphical_abstract.py  ->  figures/graphical_abstract.png
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "figures")

# Sans-serif throughout (Elsevier requirement).
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 0.8,
})

BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#555555"

# --- headline numbers (deployed results) -------------------------------------
MPC = 1.17                       # mg/L, WT MPC (Table 3)
LOG2_MPC = np.log2(MPC)          # +0.23
SD_LOG2 = (np.log2(3.78) - np.log2(0.36)) / (2 * 1.96)   # from the 95% CI ~0.87
W = 4.16                         # doublings, WT mutant-selection window
T_HALF = 6.3                     # h
T_MSW = 26.8                     # h, median population (oral-corrected)
CMAX = 17.1                      # mg/L, 3 g single oral dose
MIC_WT = MPC / 2 ** W            # lower edge of the window


def main():
    fig = plt.figure(figsize=(13.28, 5.31))
    # Three story panels with arrow gutters between them.
    gs = fig.add_gridspec(1, 3, left=0.035, right=0.985, top=0.84, bottom=0.10,
                          wspace=0.42)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    fig.suptitle("Resistance-aware MPC and the mutant-selection window as a "
                 "genotype-informed posterior",
                 fontsize=15, fontweight="bold", y=0.965)

    # =========================================================================
    # Stage 1 — genotype -> (MIC, fitness) map
    # =========================================================================
    axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    axA.text(0.5, 0.97, "1 · Genotype → (MIC, fitness) map", ha="center",
             va="top", fontsize=12.5, fontweight="bold", color=BLUE)
    chips = ["gyrA S91F", "parC D86N", "gyrB D429N"]
    for i, c in enumerate(chips):
        y = 0.78 - i * 0.135
        box = FancyBboxPatch((0.06, y - 0.045), 0.52, 0.09,
                             boxstyle="round,pad=0.012,rounding_size=0.02",
                             linewidth=1.1, edgecolor=GREY,
                             facecolor="#eef3f8")
        axA.add_patch(box)
        axA.text(0.32, y, c, ha="center", va="center", fontsize=11,
                 family="DejaVu Sans Mono")
    axA.annotate("", xy=(0.78, 0.55), xytext=(0.62, 0.55),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color=GREY))
    mapbox = FancyBboxPatch((0.70, 0.20), 0.27, 0.62,
                            boxstyle="round,pad=0.02,rounding_size=0.03",
                            linewidth=1.4, edgecolor=BLUE, facecolor="#dce8f5")
    axA.add_patch(mapbox)
    axA.text(0.835, 0.51,
             "additive\n+\nepistasis\nbasis",
             ha="center", va="center", fontsize=11, color=BLUE, fontweight="bold")
    axA.text(0.5, 0.10,
             "interventional coefficients · interval-censored, Bayesian",
             ha="center", va="center", fontsize=9.5, color=GREY, style="italic")

    # =========================================================================
    # Stage 2 — closed-form MPC posterior + PK coupling
    # =========================================================================
    axB.set_title("2 · Closed-form MPC posterior", fontsize=12.5,
                  fontweight="bold", color=BLUE, pad=8)
    g2 = np.linspace(LOG2_MPC - 4 * SD_LOG2, LOG2_MPC + 4 * SD_LOG2, 500)
    mic = 2 ** g2
    dens = np.exp(-0.5 * ((g2 - LOG2_MPC) / SD_LOG2) ** 2)
    dens /= dens.max()
    axB.fill_between(mic, dens, color=BLUE, alpha=0.30, zorder=1)
    axB.plot(mic, dens, color=BLUE, lw=1.6, zorder=2)
    # mutant-selection window: from WT MIC up to the MPC.
    axB.axvspan(MIC_WT, MPC, color=GREEN, alpha=0.10, zorder=0)
    axB.axvline(MPC, color=BLUE, lw=2, zorder=3)
    axB.axvline(CMAX, color=GREY, lw=1.6, ls="--", zorder=3)
    axB.set_xscale("log")
    axB.set_xlim(MIC_WT * 0.5, CMAX * 2.2)
    axB.set_ylim(0, 1.32)
    axB.set_yticks([])
    axB.set_xlabel("concentration (mg/L)", fontsize=10.5)
    axB.text(MPC, 1.20, f"MPC\n{MPC:.2f} mg/L", ha="center", va="bottom",
             fontsize=10.5, color=BLUE, fontweight="bold")
    axB.text(CMAX, 1.20, f"$C_{{max}}$\n{CMAX:.0f} mg/L", ha="center", va="bottom",
             fontsize=10, color=GREY)
    axB.text(np.sqrt(MIC_WT * MPC), 0.50, f"window\nW = {W:.2f}\ndoublings",
             ha="center", va="center", fontsize=9.5, color=GREEN.replace("2ca02c","1a7a1a"),
             fontweight="bold")
    axB.text(0.5, -0.30, r"$T_{\mathrm{MSW}} = t_{1/2}\times W = $"
             f" {T_MSW:.1f} h",
             transform=axB.transAxes, ha="center", va="center", fontsize=12.5,
             color="black",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff4d6", ec="#d9a400", lw=1.2))
    axB.spines[["top", "right"]].set_visible(False)

    # =========================================================================
    # Stage 3 — external validation against HFIM
    # =========================================================================
    axC.set_title("3 · Externally validated (hollow-fibre)", fontsize=12.5,
                  fontweight="bold", color=GREEN, pad=8)
    axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis("off")
    checks = [
        ("Predicted MPC 1.14 mg/L matches\nselected-mutant MIC 0.5–1.0 mg/L", GREEN),
        ("gyrB D429N effect: +4.2 (independent)\n≈ +2.9 (HFIM), to 1.5 SD", GREEN),
        ("argmax = gyrB D429N — selected in\nevery regrowth arm", GREEN),
    ]
    for i, (txt, col) in enumerate(checks):
        y = 0.86 - i * 0.205
        axC.text(0.02, y, "✓", ha="left", va="center", fontsize=16,
                 color=col, fontweight="bold")
        axC.text(0.11, y, txt, ha="left", va="center", fontsize=10)
    # punchline strip: dose ladder, peak>MPC necessary but not sufficient.
    y0 = 0.16
    axC.text(0.5, 0.305, "peak > MPC is necessary but NOT sufficient",
             ha="center", va="center", fontsize=10.5, fontweight="bold",
             color=RED)
    doses = [0.5, 1, 2, 3, 6]
    outcomes = ["select", "select", "erad", "erad", "erad"]
    x_positions = np.linspace(0.12, 0.88, len(doses))
    for x, d, o in zip(x_positions, doses, outcomes):
        col = RED if o == "select" else GREEN
        axC.scatter(x, y0, s=210, color=col, edgecolor="k", linewidth=0.6,
                    zorder=3)
        axC.text(x, y0, f"{d:g}", ha="center", va="center", fontsize=8.5,
                 color="white", fontweight="bold", zorder=4)
        axC.text(x, y0 - 0.085, "g", ha="center", va="center", fontsize=7.5,
                 color=GREY)
    axC.annotate("", xy=(x_positions[-1] + 0.05, y0), xytext=(0.06, y0),
                 arrowprops=dict(arrowstyle="-|>", lw=1.3, color=GREY), zorder=1)
    axC.text(0.5, 0.015, "single oral dose  →  ≥ 2 g eradicates  ·  < 2 g selects R",
             ha="center", va="center", fontsize=9, color=GREY, style="italic")

    # =========================================================================
    # inter-stage flow arrows (figure coordinates)
    # =========================================================================
    for x0 in (0.345, 0.665):
        arr = FancyArrowPatch((x0, 0.46), (x0 + 0.025, 0.46),
                              transform=fig.transFigure,
                              arrowstyle="-|>", mutation_scale=26,
                              lw=2.6, color="#999999")
        fig.add_artist(arr)

    out = os.path.join(FIG_DIR, "graphical_abstract.png")
    fig.savefig(out, dpi=150, facecolor="white")
    w, h = fig.get_size_inches() * 150
    print(f"Saved graphical abstract: {out}  ({int(w)} x {int(h)} px)")


if __name__ == "__main__":
    main()
