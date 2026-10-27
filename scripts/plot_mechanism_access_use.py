"""Mechanism figure: evidence access, answer quality, and residual error stages.

Panel (a): Component Recall vs Accuracy per knowledge form on the 145-record
balanced panel, with the 571-record gold-evidence ceiling as a dashed line.
Panel (b): absolute Vanilla RAG residual errors split into access-stage cases
and use-stage cases, with the use-stage share annotated.

Drawn at final print size for a single ACM column (3.33 in wide).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "rag" / "figures"

FORMS = ["Document Corpus", "Semantic KG", "Evidence Graph"]
ACCESS_COLOR = "#4C78A8"
ANSWER_COLOR = "#F58518"
ERROR_ACCESS_COLOR = "#AFC6E1"
ERROR_USE_COLOR = "#D96B3B"

CR = [0.2437, 0.2828, 0.6540]      # 145-record balanced panel
ACC = [0.3517, 0.4069, 0.4621]     # same panel, same runs
CEILING = 0.8476                    # gold-evidence ceiling, 571 records

ERR_TOTAL = [376, 334, 327]
ERR_ACCESS = [375, 288, 271]
ERR_USE = [1, 46, 56]
USE_SHARE = ["0.3%", "13.8%", "17.1%"]

plt.rcParams.update({
    "font.size": 6.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(3.33, 1.72), dpi=300, gridspec_kw={"width_ratios": [1.12, 1.22], "wspace": 0.34}
)

# ---------------- Panel (a): access vs answer ----------------
x = [0, 1, 2]
w = 0.38
bars1 = ax1.bar([i - w / 2 for i in x], CR, width=w, color=ACCESS_COLOR, edgecolor="none")
bars2 = ax1.bar([i + w / 2 for i in x], ACC, width=w, color=ANSWER_COLOR, edgecolor="none")
for i, (cr, acc) in enumerate(zip(CR, ACC)):
    ax1.text(i - w / 2, cr + 0.015, f"{cr:.2f}", ha="center", va="bottom", fontsize=5.2)
    ax1.text(i + w / 2, acc + 0.015, f"{acc:.2f}", ha="center", va="bottom", fontsize=5.2)
ax1.axhline(CEILING, ls=(0, (4, 2)), lw=0.8, color="#374151")
ax1.text(2.42, CEILING - 0.005, "gold-evidence\nceiling 0.85", ha="right", va="top", fontsize=5.2, color="#374151")
ax1.set_xticks(x)
ax1.set_xticklabels(FORMS, fontsize=5.0, rotation=18, ha="right", rotation_mode="anchor")
ax1.set_ylim(0, 1.0)
ax1.set_ylabel("Score", fontsize=6.0)
ax1.set_title("(a) Access and answer quality", fontsize=6.4, pad=3)
ax1.legend(
    [bars1[0], bars2[0]],
    ["Component Recall", "Accuracy"],
    loc="upper left", fontsize=5.0, frameon=False, handlelength=1.0, borderaxespad=0.1,
)
ax1.spines[["top", "right"]].set_visible(False)

# ---------------- Panel (b): residual error composition ----------------
bars_acc = ax2.bar(x, ERR_ACCESS, width=0.68, color=ERROR_ACCESS_COLOR, edgecolor="none")
bars_use = ax2.bar(x, ERR_USE, width=0.68, bottom=ERR_ACCESS, color=ERROR_USE_COLOR, edgecolor="white", linewidth=0.3)
for i in x:
    ax2.text(i, ERR_TOTAL[i] + 6, f"{ERR_TOTAL[i]}", ha="center", va="bottom", fontsize=5.4)
    if ERR_USE[i] > 8:
        ax2.text(i, ERR_ACCESS[i] + ERR_USE[i] / 2, USE_SHARE[i], ha="center", va="center",
                 fontsize=5.0, color="white", fontweight="bold")
    else:
        ax2.text(i, ERR_ACCESS[i] - 28, USE_SHARE[i], ha="center", va="center",
                 fontsize=5.0, color="#374151", fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(FORMS, fontsize=5.0, rotation=18, ha="right", rotation_mode="anchor")
ax2.set_xlim(-0.58, 2.58)
ax2.set_ylim(0, 455)
ax2.set_ylabel("Residual errors", fontsize=6.0)
ax2.set_title("(b) Residual error mix", fontsize=6.4, pad=3)
ax2.legend(
    [Patch(facecolor=ERROR_ACCESS_COLOR), Patch(facecolor=ERROR_USE_COLOR)],
    ["access stage", "use stage (share)"],
    loc="upper right", ncol=1, fontsize=4.8, frameon=False,
    handlelength=0.9, labelspacing=0.25, borderaxespad=0.15,
)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout(pad=0.4)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"mechanism_access_use.{ext}", bbox_inches="tight")
print("written", OUT / "mechanism_access_use.pdf")
