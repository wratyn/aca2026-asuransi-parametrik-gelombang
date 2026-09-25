#!/usr/bin/env python3
"""fleet exposure per province x tier, province->zone weights"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

AKAR = Path(__file__).resolve().parent.parent
RAW = AKAR / "data" / "raw"
PROV = ["BANTEN", "JAWA BARAT", "JAWA TENGAH", "DAERAH ISTIMEWA YOGYAKARTA",
        "JAWA TIMUR", "BALI", "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR"]
TIER = {1: ["MOTOR_TEMPEL", "KM_LT5"], 2: ["KM_5_10", "KM_10_30"], 3: ["KM_GT30"]}
THN = [str(y) for y in range(2019, 2025)]

k = pd.read_csv(RAW / "kkp_kapal" / "kapal_provinsi_2019_2024.csv")
k = k[k.provinsi.isin(PROV)].set_index(["tabel", "provinsi"])[THN]
rows = []
for p in PROV:
    for t, tabs in TIER.items():
        sub = k.loc[[(tb, p) for tb in tabs if (tb, p) in k.index]]
        med = sum(np.nan_to_num(np.nanmedian(r[["2022", "2023", "2024"]].to_numpy(float)))
                  if r[["2022", "2023", "2024"]].notna().any() else 0.0 for _, r in sub.iterrows())
        rows.append(dict(provinsi=p, tier=t, n_kapal_median_2022_2024=int(round(med)),
                         **{f"n_{y}": int(sub[y].fillna(0).sum()) for y in THN}))
eks = pd.DataFrame(rows)
eks.to_csv(AKAR / "data/processed/eksposur_provinsi_gt.csv", index=False)

fs = sorted(glob.glob(str(RAW / "kkp_produksi" / "produksi_tangkap_laut_*.csv")))
d = pd.concat([pd.read_csv(f, low_memory=False,
                           usecols=["provinsi", "wpp", "produksi", "nilai_produksi_rp"]) for f in fs])
d = d[d.provinsi.isin(PROV)].copy()
for c in ("produksi", "nilai_produksi_rp"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["di573"] = d.wpp == "WPP-RI-573"
tot = d.groupby("provinsi")[["nilai_produksi_rp", "produksi"]].sum()
di = d[d.di573].groupby("provinsi")[["nilai_produksi_rp", "produksi"]].sum()
s_nil = (di.nilai_produksi_rp / tot.nilai_produksi_rp).reindex(PROV)
s_vol = (di.produksi / tot.produksi).reindex(PROV)

DASAR = "pangsa nilai produksi tangkap laut prov ke WPP-RI-573, KKP 2019-2024"
NTT_D, NTT_F = 26 / 45, 19 / 45
br = []
for p in PROV:
    if p == "NUSA TENGGARA TIMUR":
        dasar = "pangsa nilai produksi ke WPP-573 x proporsi sel pesisir zona (26:19) — asumsi A9b"
        br.append(dict(provinsi=p, wpp="WPP-RI-573", zona="573-D", bobot=round(s_nil[p] * NTT_D, 4),
                       bobot_alt_volume=round(s_vol[p] * NTT_D, 4), dasar_bobot=dasar))
        br.append(dict(provinsi=p, wpp="WPP-RI-573", zona="573-F", bobot=round(s_nil[p] * NTT_F, 4),
                       bobot_alt_volume=round(s_vol[p] * NTT_F, 4), dasar_bobot=dasar))
    else:
        br.append(dict(provinsi=p, wpp="WPP-RI-573", zona="573-A", bobot=round(s_nil[p], 4),
                       bobot_alt_volume=round(s_vol[p], 4), dasar_bobot=DASAR))
for p in PROV:
    br.append(dict(provinsi=p, wpp="LUAR CAKUPAN", zona="-", bobot=round(1 - s_nil[p], 4),
                   bobot_alt_volume=round(1 - s_vol[p], 4),
                   dasar_bobot="sisa: WPP 572/711/712/713/714/717/718"))
pd.DataFrame(br).to_csv(AKAR / "data/dim/bridge_provinsi_zona.csv", index=False)

e = eks[["provinsi", "tier", "n_kapal_median_2022_2024"]].rename(columns={"n_kapal_median_2022_2024": "n_kapal_prov"})
e["zona"] = np.where(e.provinsi == "NUSA TENGGARA TIMUR", "573-E", "573-B")  # old zone labels (B/E), mapped downstream
e["bobot"] = e.provinsi.map(s_nil.round(4))
e["n_kapal_zona"] = (e.n_kapal_prov * e.bobot).round(1)
e["n_kapal_zona_alt_volume"] = (e.n_kapal_prov * e.provinsi.map(s_vol.round(4))).round(1)
e = e[["provinsi", "zona", "tier", "n_kapal_prov", "bobot", "n_kapal_zona", "n_kapal_zona_alt_volume"]]
e.to_csv(AKAR / "data/processed/eksposur.csv", index=False)

print(eks.to_string(index=False))
print(f"\nTotal eksposur tertimbang WPP 573: {e.n_kapal_zona.sum():,.0f} kapal")
print("-> eksposur_provinsi_gt.csv, bridge_provinsi_zona.csv, eksposur.csv")
