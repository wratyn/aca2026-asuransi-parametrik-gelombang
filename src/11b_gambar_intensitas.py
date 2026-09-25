#!/usr/bin/env python3
"""fig 3.3, seasonal intensity"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

AKAR = Path(__file__).resolve().parent.parent
T = 365.25
SEL = [("573-A", 1, "<5 GT", "#1f4e79"), ("573-A", 2, "5-30 GT", "#2e8b57"), ("573-F", 1, "<5 GT", "#c1440e")]
BATAS = [(0, "Barat"), (90, "Peralihan I"), (152, "Timur"), (244, "Peralihan II")]

fl = pd.read_csv(AKAR / "output/tables/fit_lambda.csv")
of = pd.read_csv(AKAR / "output/tables/optimasi_final.csv")
d = np.linspace(0, 365, 800)

fig, ax = plt.subplots(figsize=(9, 3.6))
for z, k, gt, warna in SEL:
    r = fl[(fl.zona == z) & (fl.tier == k)].iloc[0]
    s = d + 12 * T
    v = r.a0 + sum(r[f"a{j}"] * np.cos(2*np.pi*j*s/T) + r[f"b{j}"] * np.sin(2*np.pi*j*s/T)
                   for j in range(1, int(r.J) + 1))
    lam_bulan = np.exp(v + r.gamma * s / 365.25) * T / 12
    u = float(of[(of.zona == z) & (of.tier == k)].u_opt.iloc[0])
    ax.plot(d, lam_bulan, color=warna, lw=2.2,
            label=f"{z} tier {k}  ({gt}, u = {u:.2f} m)".replace(".", ","))
ytop = 3.5
for x, nama in BATAS:
    if x > 0:
        ax.axvline(x, color="#c5c5c5", lw=1)
    ax.text(x + 4, ytop, nama, color="#737373", fontsize=8.5, va="top")
ax.set_xlim(0, 365); ax.set_ylim(top=ytop)
ax.set_xlabel("hari ke- dalam tahun"); ax.set_ylabel("episode per bulan")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="center left")
fig.tight_layout()
fig.savefig(AKAR / "output/figures/fig_intensitas_final.png", dpi=160, facecolor="white")
print("selesai: output/figures/fig_intensitas_final.png")
