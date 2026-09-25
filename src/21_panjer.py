#!/usr/bin/env python3
"""Panjer recursion, Poisson counts"""
from pathlib import Path
import numpy as np, pandas as pd
AKAR = Path(__file__).resolve().parent.parent
T, M_K, L, Q = 365.25, 7, 20, 0.25
KMAKS = 400
MUSIM = ['Barat','Peralihan I','Timur','Peralihan II']
BULAN = {12:'Barat',1:'Barat',2:'Barat',3:'Peralihan I',4:'Peralihan I',5:'Peralihan I',
         6:'Timur',7:'Timur',8:'Timur',9:'Peralihan II',10:'Peralihan II',11:'Peralihan II'}
DIJUAL = {("573-A",1), ("573-A",2), ("573-F",1)}

FL = pd.read_csv(AKAR/"output/tables/fit_lambda.csv")
PM = pd.read_csv(AKAR/"data/processed/pmf_durasi.csv")
BK = pd.read_csv(AKAR/"output/tables/b_k_usulan.csv")
EP = pd.read_csv(AKAR/"data/processed/episode.csv")
def bk_row(z, k):
    r = BK[(BK.zona == ("573-A" if z == "573-A" else "573-DF")) & (BK.tier == k)]
    return r.iloc[0] if len(r) else None

tgl = pd.date_range("2001-01-01", "2025-12-31", freq="D")
t_abs = (tgl - pd.Timestamp("2001-01-01")).days.to_numpy().astype(float)
mus_h = np.array([BULAN[m] for m in tgl.month])
thm_h = np.where(tgl.month == 12, tgl.year+1, tgl.year)

def intensitas(r):
    J = int(r.J); v = np.full(len(t_abs), r.a0)
    for j in range(1, J+1):
        v += r[f"a{j}"]*np.cos(2*np.pi*j*t_abs/T) + r[f"b{j}"]*np.sin(2*np.pi*j*t_abs/T)
    return np.exp(v + r.gamma*t_abs/365.25)

def panjer_poisson(lam, fD, K=KMAKS):
    g = np.zeros(K+1); g[0] = np.exp(-lam)
    for x in range(1, K+1):
        s = sum(y*fD[y]*g[x-y] for y in range(1, min(x, len(fD)-1)+1))
        g[x] = lam/x*s
    return g

def kuantil(g, q):
    c = np.cumsum(g)
    return int(np.searchsorted(c, q))

rows_dist, rows_val, rows_prem = [], [], []
for _, r in FL.iterrows():
    z, k = r.zona, int(r.tier)
    lam_h = intensitas(r)
    pm = PM[(PM.zona == z) & (PM.tier == k)].sort_values("d")
    for kolom, label in (("prob_empiris","empiris"), ("prob","parametrik")):
        fD = np.zeros(M_K+1); fD[1:] = pm[kolom].to_numpy()
        fD[1:] /= fD[1:].sum()
        for s in MUSIM:
            lam_th = np.array([lam_h[(mus_h == s) & (thm_h == y)].sum()
                               for y in sorted(set(thm_h[mus_h == s]))])
            lam_th = lam_th[lam_th > 0]
            if s == 'Barat': lam_th = lam_th[1:]
            lam_s = lam_th.mean()
            g = panjer_poisson(lam_s, fD)
            x = np.arange(len(g))
            f = kuantil(g, Q)
            P = np.minimum(np.clip(x - f, 0, None), L)
            EW, EP_ = float((x*g).sum()), float((P*g).sum())
            p_bayar = float(g[x > f].sum())
            if label == "empiris":
                rows_dist.append(dict(zona=z, tier=k, musim=s, lambda_musim=lam_s,
                                      massa=g.sum(), E_W=EW, sd_W=float(np.sqrt((x**2*g).sum()-EW**2)),
                                      P_W0=float(g[0]), f_panjer=f, E_payout=EP_, p_bayar=p_bayar,
                                      W_q90=kuantil(g, 0.90), W_q99=kuantil(g, 0.99)))
            rows_prem.append(dict(zona=z, tier=k, musim=s, severitas=label, lambda_musim=lam_s,
                                  f=f, E_payout_hari=EP_, p_bayar=p_bayar))
    e = EP[(EP.zona == z) & (EP.tier == k)].copy()
    e["W"] = np.minimum(e.durasi, M_K)
    obs = e.groupby(["musim","tahun"]).W.sum()
    for s in MUSIM:
        v = obs.loc[s] if s in obs.index.get_level_values(0) else pd.Series(dtype=float)
        d = [q for q in rows_dist if q["zona"] == z and q["tier"] == k and q["musim"] == s][0]
        rows_val.append(dict(zona=z, tier=k, musim=s, n_musim=len(v),
                             E_W_model=d["E_W"], E_W_data=float(v.mean()),
                             sd_model=d["sd_W"], sd_data=float(v.std(ddof=1)),
                             massa_panjer=d["massa"]))

D = pd.DataFrame(rows_dist); D.to_csv(AKAR/"output/tables/dist_panjer.csv", index=False)
V = pd.DataFrame(rows_val);  V.to_csv(AKAR/"output/tables/validasi_panjer.csv", index=False)
PR = pd.DataFrame(rows_prem)
PR["b_k"] = [float(bk_row(z,k).b_pakai) if bk_row(z,k) is not None else np.nan
             for z,k in zip(PR.zona, PR.tier)]
PR["premi_musim"] = PR.E_payout_hari*PR.b_k
PR["dijual"] = [(z,k) in DIJUAL for z,k in zip(PR.zona, PR.tier)]
PR.to_csv(AKAR/"output/tables/premi_panjer.csv", index=False)

pd.set_option("display.width", 260)
print("== massa total Panjer (harus 1) ==")
print(f"min {D.massa.min():.10f}  maks {D.massa.max():.10f}")
print("\n== validasi E[W] dan sd: model vs data ==")
print(V.round(2).to_string(index=False))
print("\n== distribusi & franchise Panjer (severitas empiris) ==")
print(D.round(3).to_string(index=False))
print("\n== premi murni per tahun, severitas empiris vs parametrik (sel S2) ==")
t = (PR[PR.dijual].groupby(["zona","tier","severitas"])
       .agg(premi_thn=("premi_musim","sum"), hari_thn=("E_payout_hari","sum")).reset_index())
print(t.round(0).to_string(index=False))
