#!/usr/bin/env python3
"""dim_tier + dim_musim"""
from pathlib import Path
import pandas as pd

AKAR = Path(__file__).resolve().parent.parent
KUNCI = {"573-A": 1, "573-D": 4, "573-F": 6}
TIER = [(1, "< 5 GT (termasuk perahu motor tempel)", 0, 5, "KM_0005;MT_*", 85),
        (2, "5 - 30 GT", 5, 30, "KM_0005_0010;KM_0010_0020;KM_0020_0030", 90),
        (3, "> 30 GT", 30, None, "KM_0030_0050;KM_0050_0100;KM_0100_0200;KM_0200_0300;KM_0300_0500", 95)]
MUSIM = [("Barat", (12, 1, 2), 90), ("Peralihan I", (3, 4, 5), 92),
         ("Timur", (6, 7, 8), 92), ("Peralihan II", (9, 10, 11), 91)]

dz = pd.read_csv(AKAR / "data/dim/dim_zona.csv").set_index("zona")
rows = []
for z in KUNCI:
    for t, lab, g0, g1, kode, p in TIER:
        rows.append(dict(zona=z, tier=t, label_tier=lab, gt_min=g0, gt_max=g1, kode_kkp=kode,
                         persentil=p, u_k_awal=dz.loc[z, f"u_p{p}"], b_k=None, m_k=7, M_k=20,
                         catatan="u_k_awal = persentil iklim lokal; final dari maksimasi HE (Tahap 7)"))
dt = pd.DataFrame(rows)
dt["gt_max"] = dt.gt_max.astype("Int64")
dt.to_csv(AKAR / "data/dim/dim_tier.csv", index=False)

sk = pd.read_csv(AKAR / "output/tables/siklus_musiman_zona_wpp573.csv")
rows = []
for z, zn in KUNCI.items():
    s = sk[sk.zona == zn].set_index("bulan").swh_ternormalisasi
    rel = {m: round(float(s.loc[list(b)].mean()), 4) for m, b, _ in MUSIM}
    puncak = max(rel, key=rel.get)
    for i, (m, _, hari) in enumerate(MUSIM, 1):
        rows.append(dict(zona=z, musim=m, urutan=i, hari_musim=hari,
                         indeks_swh_relatif=rel[m], musim_risiko_tinggi=int(m == puncak)))
pd.DataFrame(rows).to_csv(AKAR / "data/dim/dim_musim.csv", index=False)
print(dt[["zona", "tier", "persentil", "u_k_awal"]].to_string(index=False))
print("-> dim_tier.csv, dim_musim.csv")
