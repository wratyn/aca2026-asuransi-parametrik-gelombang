#!/usr/bin/env python3
"""lock design (20 Sep): sold cells, net-margin b_k"""
from pathlib import Path
import numpy as np, pandas as pd
AKAR = Path(__file__).resolve().parent.parent
T, DIM = AKAR/"output/tables", AKAR/"data/dim"

DIJUAL = {("573-A",1), ("573-A",2), ("573-A",3), ("573-F",1)}
HE_BATAS = 0.40
MARGIN = "bersih"

bk = pd.read_csv(T/"b_k_usulan.csv")
bk["basis_margin"] = MARGIN
bk["b_pakai"] = bk.b_bersih
bk.to_csv(T/"b_k_usulan.csv", index=False)
def row_bk(z, k):
    r = bk[(bk.zona == ("573-A" if z == "573-A" else "573-DF")) & (bk.tier == k)]
    return r.iloc[0] if len(r) else None
def b_col(zs, ks, kol):
    return [row_bk(z, k)[kol] if row_bk(z, k) is not None else np.nan for z, k in zip(zs, ks)]

D = pd.read_csv(DIM/"dim_desain_polis.csv")
D = D.drop(columns=[c for c in ("b_k_kontribusi", "dijual") if c in D])
D["b_k"] = b_col(D.zona, D.tier, "b_pakai")
D["premi_murni_musim"] = D.hari_bayar_rata * D.b_k
D["b_k_kontribusi"] = b_col(D.zona, D.tier, "b_base")
D["dijual"] = [(z, k) in DIJUAL for z, k in zip(D.zona, D.tier)]
D.to_csv(DIM/"dim_desain_polis.csv", index=False)

F = pd.read_csv(T/"optimasi_final.csv")
F = F.drop(columns=[c for c in ("premi_murni_thn_kontribusi", "dijual") if c in F])
F["premi_murni_thn"] = F.hari_bayar_thn * np.array(b_col(F.zona, F.tier, "b_pakai"))
F["layak"] = F.HE_p10_halus >= HE_BATAS
F["premi_murni_thn_kontribusi"] = F.hari_bayar_thn * np.array(b_col(F.zona, F.tier, "b_base"))
F["dijual"] = [(z, k) in DIJUAL for z, k in zip(F.zona, F.tier)]
assert (F.dijual <= F.layak).all(), "ada sel dijual yang tidak lolos batas HE"
F.to_csv(T/"optimasi_final.csv", index=False)

E = pd.read_csv(AKAR/"data/processed/eksposur.csv")
eks = E.groupby(["zona","tier"]).n_kapal_zona.sum()
n_sel = pd.read_csv(DIM/"dim_zona.csv").set_index("zona").n_sel
porsi_F = n_sel["573-F"] / (n_sel["573-D"] + n_sel["573-F"])
def eksposur(z, k):
    if z == "573-A": return eks.get(("573-B", k), 0.0)
    tot = eks.get(("573-E", k), 0.0)
    return tot*porsi_F if z == "573-F" else tot*(1-porsi_F)

rows = []
for _, r in F.iterrows():
    b = row_bk(r.zona, r.tier)
    if b is None: continue
    n = eksposur(r.zona, r.tier)
    rows.append(dict(zona=r.zona, tier=r.tier, dijual=r.dijual, HE_p10_halus=round(r.HE_p10_halus,3),
        hari_bayar_thn=round(r.hari_bayar_thn,2), p_bayar=round(r.p_bayar,3),
        b_k_lama=b.b_base, b_k_baru=b.b_pakai,
        premi_lama=r.premi_murni_thn_kontribusi, premi_baru=r.premi_murni_thn,
        perubahan_pct=100*(r.premi_murni_thn/r.premi_murni_thn_kontribusi-1),
        premi_bulan=r.premi_murni_thn/12, pct_R=100*r.premi_murni_thn/b.R_med,
        hari_pendapatan_per_bulan=(r.premi_murni_thn/12)/(b.R_med/b.D_base),
        manfaat_maks_thn=4*20*b.b_pakai, eksposur_kapal=round(n),
        volume_premi_M=n*r.premi_murni_thn/1e9))
R = pd.DataFrame(rows); R.to_csv(T/"premi_murni_kunci.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(2).to_string())
J = R[R.dijual]
print(f"\nporsi_F (proksi) = {porsi_F:.3f}")
print(f"Eksposur dijual: {J.eksposur_kapal.sum():,.0f} dari {R.eksposur_kapal.sum()+ eks.get(('573-E',3),0):,.0f} kapal")
print(f"Volume premi murni dijual: Rp{J.volume_premi_M.sum():,.1f} M/thn; seandainya b lama: "
      f"Rp{(J.eksposur_kapal*J.premi_lama).sum()/1e9:,.1f} M")
