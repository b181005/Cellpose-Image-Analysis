"""
compare_channels.py

Displays a grid comparing ch1 and ch2 images for DMT and Saline samples.
Layout: each row = one sample, columns = [ch1, ch2]
DMT samples shown first, then Saline samples.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

BASE_DIR = Path(r"C:\Users\b1810\OneDrive - zonemail.clpccd.edu\abrain\LAB\10-20-25_Confocol\converted_images\Converted_2026-03-24")

def collect_pairs(folder: Path, treatment: str):
    """Return list of (label, ch1_path, ch2_path) for each matched pair."""
    pairs = []
    ch1_files = sorted(folder.glob("*_ch1.jpg"))
    for ch1 in ch1_files:
        base = ch1.name.replace("_ch1.jpg", "")
        ch2 = folder / (base + "_ch2.jpg")
        if ch2.exists():
            label = f"{treatment}\n{base}"
            pairs.append((label, ch1, ch2))
    return pairs

dmt_pairs   = collect_pairs(BASE_DIR / "DMT",    "DMT")
saline_pairs = collect_pairs(BASE_DIR / "Saline", "Saline")
all_pairs = dmt_pairs + saline_pairs

if not all_pairs:
    print("No matching ch1/ch2 pairs found. Check the folder paths.")
    exit()

# ── Layout ────────────────────────────────────────────────────────────────────
# Two columns per treatment side-by-side: [DMT ch1 | DMT ch2 | Saline ch1 | Saline ch2]
# Rows = max(len(dmt_pairs), len(saline_pairs))

n_dmt    = len(dmt_pairs)
n_saline = len(saline_pairs)
n_rows   = max(n_dmt, n_saline)
n_cols   = 4  # dmt_ch1 | dmt_ch2 | saline_ch1 | saline_ch2

FIG_W = 20
FIG_H = max(2 * n_rows, 6)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W, FIG_H))
fig.suptitle("Channel Comparison: DMT vs Saline\n(ch1 = red, ch2 = green)",
             fontsize=14, fontweight="bold", y=1.005)

# Column headers
col_titles = ["DMT — ch1", "DMT — ch2", "Saline — ch1", "Saline — ch2"]
for col, title in enumerate(col_titles):
    ax = axes[0, col] if n_rows > 1 else axes[col]
    ax.set_title(title, fontsize=11, fontweight="bold", color="white",
                 backgroundcolor=("#8B0000" if "DMT" in title else "#00447c"),
                 pad=6)

def to_grayscale(img):
    """Convert RGB image to 2D grayscale array."""
    import numpy as np
    if img.ndim == 3:
        return img[..., :3].mean(axis=2)
    return img

def plot_pair(row, left_col, pair):
    import numpy as np
    label, ch1_path, ch2_path = pair
    cmaps = ["Reds", "Greens"]
    for offset, (img_path, cmap) in enumerate(zip([ch1_path, ch2_path], cmaps)):
        ax = axes[row, left_col + offset] if n_rows > 1 else axes[left_col + offset]
        img = mpimg.imread(img_path)
        gray = to_grayscale(img)
        ax.imshow(gray, cmap=cmap, vmin=0, vmax=255)
        ax.axis("off")
        if offset == 0:
            sample_name = label.split("\n", 1)[-1]
            ax.set_ylabel(sample_name, fontsize=7, rotation=0,
                          labelpad=60, va="center", ha="right")

# Plot DMT pairs (cols 0–1)
for row, pair in enumerate(dmt_pairs):
    plot_pair(row, 0, pair)

# Plot Saline pairs (cols 2–3)
for row, pair in enumerate(saline_pairs):
    plot_pair(row, 2, pair)

# Hide unused axes
for row in range(n_dmt, n_rows):
    for col in [0, 1]:
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis("off")
for row in range(n_saline, n_rows):
    for col in [2, 3]:
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis("off")

plt.tight_layout(h_pad=0.3, w_pad=0.3)

# Save and show
out_path = BASE_DIR / "RESULTS" / "channel_comparison_dmt_vs_saline.png"
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
print(f"DMT pairs: {n_dmt}  |  Saline pairs: {n_saline}")

plt.show()
