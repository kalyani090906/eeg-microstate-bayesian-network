"""
make_kim_style_tables.py - the HC vs SC group comparison written in the SAME LAYOUT as Table 2 of
Kim et al. 2021 [9] (PLOS ONE, PMC8121321), as asked by the faculty.

Kim et al. Table 2 layout (copied):
  columns  Type A | Type B | Type C | Type D | Across types
  blocks   Duration (ms), Occurrence (Hz), Coverage (%)
  rows     SC mean (±SD), HC mean (±SD), Cohen's d, p-value, t-value; significant p in bold
  "Across types": mean duration (= all labelled time / all segments = 1000 / total occurrence) and
  mean occurrence (= segments per second, all states together).
Differences from Kim et al. (stated under each table):
  * one value per SUBJECT (14 SC vs 14 HC). Kim et al.'s numbers appear to be per 5-s epoch
    ("total time 4.551 s" per row), which gives far smaller p-values because epochs of one
    person are not independent.
  * p-value = Wilcoxon rank-sum test (asked by the faculty), not corrected; t-value from an
    independent t-test is also given, as in Kim et al.
  * Mean GFP is not included (not computed in this project).
  * A second table gives the 8 transition probabilities of the 16-feature set in the same layout.

Input:  outputs/eeg_features_epoch_mean_15min.csv                 (shared maps, 24 features)
        outputs/group_specific/features_group_specific_24.csv     (separate HC / SC maps)
Output (outputs/group_specific/):
        kim_style_table_<maps>.csv  and  kim_style_table_<maps>.png    (<maps> = group_maps, shared_maps)

Run:  eeg_env\\Scripts\\python.exe scripts\\make_kim_style_tables.py
"""

# ---------------- settings (change only here) ----------------
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUT = OUTPUT_DIR / "group_specific"
TABLES = {"group_maps": ("separate HC / SC maps", OUT / "features_group_specific_24.csv"),
          "shared_maps": ("common maps (all 28)", OUTPUT_DIR / "eeg_features_epoch_mean_15min.csv")}
ALPHA = 0.05
TRANSITIONS = ["trans_0to1", "trans_0to2", "trans_1to0", "trans_1to2",       # the 8 of the 16-feature set
               "trans_2to0", "trans_2to1", "trans_3to0", "trans_3to1"]
# --------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402
from scipy.stats import ranksums, ttest_ind  # noqa: E402

import microstate_model as mm               # noqa: E402

LETTER_TO_STATE = {v: k for k, v in mm.STATE_NAMES.items()}     # A -> 0, B -> 2, C -> 3, D -> 1
TYPES = ["A", "B", "C", "D"]


def compare(sc, hc):
    # Kim-style numbers for one feature: means, SDs, Cohen's d, p (Wilcoxon rank-sum), t
    pooled_sd = np.sqrt(((len(sc) - 1) * sc.var(ddof=1) + (len(hc) - 1) * hc.var(ddof=1)) / (len(sc) + len(hc) - 2))
    return {"SC": f"{sc.mean():.2f} (±{sc.std(ddof=1):.2f})", "HC": f"{hc.mean():.2f} (±{hc.std(ddof=1):.2f})",
            "Cohen's d": f"{(sc.mean() - hc.mean()) / pooled_sd:.2f}",
            "p-value": ranksums(sc, hc).pvalue, "t-value": f"{ttest_ind(sc, hc).statistic:.3f}"}


for key, (title, path) in TABLES.items():
    df = pd.read_csv(path)
    sc, hc = df[df.label == 1], df[df.label == 0]
    occ_cols = [f"state{LETTER_TO_STATE[t]}_occurrence" for t in TYPES]

    # Step 1: blocks of Kim's Table 2 (values per subject)
    blocks = {}
    for measure, what, scale in [("Duration (ms)", "duration", 1000), ("Occurrence (Hz)", "occurrence", 1),
                                 ("Coverage (%)", "coverage", 100)]:
        cols = {t: compare(sc[f"state{LETTER_TO_STATE[t]}_{what}"].to_numpy() * scale,
                           hc[f"state{LETTER_TO_STATE[t]}_{what}"].to_numpy() * scale) for t in TYPES}
        if what == "duration":       # across types: 1000 / total occurrence = mean segment length
            cols["Across types"] = compare(1000 / sc[occ_cols].sum(axis=1).to_numpy(),
                                           1000 / hc[occ_cols].sum(axis=1).to_numpy())
        elif what == "occurrence":   # across types: segments per second, all states
            cols["Across types"] = compare(sc[occ_cols].sum(axis=1).to_numpy(), hc[occ_cols].sum(axis=1).to_numpy())
        blocks[measure] = cols
    trans = {}
    for c in TRANSITIONS:
        i, j = c[6:].split("to")
        trans[f"{mm.STATE_NAMES[int(i)]}->{mm.STATE_NAMES[int(j)]}"] = compare(sc[c].to_numpy(), hc[c].to_numpy())

    # Step 2: one long CSV (block, row, columns)
    rows = []
    for block, cols in list(blocks.items()) + [("Transition probability", trans)]:
        for r in ["SC", "HC", "Cohen's d", "p-value", "t-value"]:
            row = {"measure": block, "row": r}
            for name, vals in cols.items():
                v = vals[r]
                row[name] = f"{v:.3f}" if r == "p-value" else v
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / f"kim_style_table_{key}.csv", index=False)

    # Step 3: picture of the two tables (significant p in bold red)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.2), gridspec_kw={"height_ratios": [16.5, 6.5]})
    for ax, (cols, block_names) in zip(axes, [(None, list(blocks)), (trans, ["Transition probability"])]):
        ax.axis("off")
        header = ["", ""] + (TYPES + ["Across types"] if cols is None else list(trans))
        cell_text, bold = [], []
        for block in block_names:
            data = blocks[block] if cols is None else trans
            for k, r in enumerate(["SC", "HC", "Cohen's d", "p-value", "t-value"]):
                line = [block if k == 0 else "", r]
                flags = [False, False]
                for name in header[2:]:
                    if name not in data:
                        line.append("-")
                        flags.append(False)
                        continue
                    v = data[name][r]
                    line.append(("<0.001" if v < 0.001 else f"{v:.3f}") if r == "p-value" else v)
                    flags.append(r == "p-value" and v < ALPHA)
                cell_text.append(line)
                bold.append(flags)
        table = ax.table(cellText=cell_text, colLabels=header, loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.45)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if r == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#eef3fb")
            elif c == 0:
                cell.set_text_props(fontweight="bold", ha="left")
            elif bold[r - 1][c]:
                cell.set_text_props(fontweight="bold", color="#b00020")
        table.auto_set_column_width(list(range(len(header))))
    axes[0].set_title(f"Microstate features, SC (n = 14) vs HC (n = 14), {title} - layout of Kim et al. [9] Table 2",
                      fontsize=12, fontweight="bold", loc="left")
    axes[1].set_title("Transition probabilities (8 of the 16 features), same layout", fontsize=11,
                      fontweight="bold", loc="left")
    fig.text(0.01, 0.01, "Values: mean (±SD) of one value per subject. p-value: Wilcoxon rank-sum test, uncorrected; "
             "bold red = p < 0.05. t-value: independent t-test.\nCohen's d: (SC mean - HC mean) / pooled SD; "
             "+ = higher in SC. Across types: mean duration = 1000 / total occurrence; mean occurrence = all "
             "segments per second.", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / f"kim_style_table_{key}.png", dpi=160)
    plt.close(fig)
    print(f"Saved kim_style_table_{key}.csv / .png")
