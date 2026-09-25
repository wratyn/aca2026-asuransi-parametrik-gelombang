#!/usr/bin/env python3
"""compare product scenarios"""
from pathlib import Path
import numpy as np, pandas as pd
AKAR = Path(__file__).resolve().parent.parent

JEDA = 2
BULAN_MUSIM = {12:"Barat",1:"Barat",2:"Barat",3:"Peralihan I",4:"Peralihan I",5:"Peralihan I",
               6:"Timur",7:"Timur",8:"Timur",9:"Peralihan II",10:"Peralihan II",11:"Peralihan II"}
def episode(x, u, jeda=JEDA):
    atas = x > u
    if not atas.any(): return np.array([]), np.array([])
    d = np.diff(atas.astype(int))
    mulai = list(np.where(d == 1)[0] + 1); akhir = list(np.where(d == -1)[0])
    if atas[0]:  mulai.insert(0, 0)
    if atas[-1]: akhir.append(len(atas) - 1)
    m, a = [mulai[0]], []
    for i in range(1, len(mulai)):
        if mulai[i] - akhir[i-1] - 1 < jeda: continue
        a.append(akhir[i-1]); m.append(mulai[i])
    a.append(akhir[-1])
    return np.array(m), np.array(a)

MUSIM = ['Barat','Peralihan I','Timur','Peralihan II']
M_K, L = 7, 20
PK = {1:85, 2:90, 3:95}
rng = np.random.default_rng(20260921)

S = pd.read_csv(AKAR/"data/processed/swh_harian_zona.csv", parse_dates=["tanggal"])
G = pd.read_csv(AKAR/"output/tables/optimasi_grid.csv")
BK = pd.read_csv(AKAR/"output/tables/b_k_usulan.csv")
E = pd.read_csv(AKAR/"data/processed/eksposur.csv")
NSEL = pd.read_csv(AKAR/"data/dim/dim_zona.csv").set_index("zona").n_sel
eks = E.groupby(["zona","tier"]).n_kapal_zona.sum()
porsi_F = NSEL["573-F"]/(NSEL["573-D"]+NSEL["573-F"])
def eksposur(z, k):
    if z == "573-A": return float(eks.get(("573-B", k), 0.0))
    t = float(eks.get(("573-E", k), 0.0))
    return t*porsi_F if z == "573-F" else t*(1-porsi_F)
def bk_row(z, k):
    r = BK[(BK.zona == ("573-A" if z == "573-A" else "573-DF")) & (BK.tier == k)]
    return r.iloc[0] if len(r) else None

Z = {}
for z, g in S.groupby("zona"):
    g = g.sort_values("tanggal")
    t = g.tanggal
    Z[z] = dict(x=g.swh_rerata.to_numpy(), mus=np.array([BULAN_MUSIM[m] for m in t.dt.month]),
                thm=np.where(t.dt.month == 12, t.dt.year+1, t.dt.year))
KOL = [(s, y) for s in MUSIM for y in range(2001, 2026) if not (s == 'Barat' and y == 2001)]
KOLI = {k:i for i,k in enumerate(KOL)}
S_OF = np.array([MUSIM.index(s) for s,_ in KOL]); TH_OF = np.array([y for _,y in KOL])

def W_musim(z, u):
    d = Z[z]; out = np.zeros(len(KOL))
    m, a = episode(d["x"], u)
    for s0, s1 in zip(m, a):
        k = (d["mus"][s0], d["thm"][s0])
        if k in KOLI: out[KOLI[k]] += min(s1-s0+1, M_K)
    return out

def payout(z, k, q, p):
    u = float(G[(G.zona == z) & (G.tier == k) & (G.p == p) & (G.q == q)].u_z.iloc[0])
    W = W_musim(z, u)
    f = np.zeros(len(W))
    if q > 0:
        for s in range(4):
            sel = S_OF == s; f[sel] = np.floor(np.quantile(W[sel], q))
    return np.minimum(np.clip(W - f, 0, None), L), u

def he_p10(z, k, q, p):
    r = G[(G.zona == z) & (G.tier == k) & (G.p == p) & (G.q == q)]
    return float(r.HE_p10.iloc[0]), float(r.p_bayar.iloc[0])

P_OPT = {("573-A",1):85, ("573-A",2):90, ("573-A",3):95, ("573-D",1):91, ("573-D",2):92,
         ("573-F",1):86, ("573-F",2):88}

SKENARIO = {
 "S0 lama tanpa franchise": dict(sel=[("573-A",1),("573-A",2),("573-A",3),("573-F",1)], q={}, qd=0.0),
 "S1 baseline 20 Sep":      dict(sel=[("573-A",1),("573-A",2),("573-A",3),("573-F",1)], q={}, qd=0.25),
 "S2 tanpa tier 3":         dict(sel=[("573-A",1),("573-A",2),("573-F",1)], q={}, qd=0.25),
 "S3 tanpa t3 + q median":  dict(sel=[("573-A",1),("573-A",2),("573-F",1)], q={}, qd=0.5),
 "S4 hanya 573-A":          dict(sel=[("573-A",1),("573-A",2)], q={}, qd=0.25),
 "S5 semua sel berpremi":   dict(sel=[("573-A",1),("573-A",2),("573-A",3),("573-F",1),("573-F",2),
                                      ("573-D",1),("573-D",2)], q={}, qd=0.25),
 "S6 tier 1 saja":          dict(sel=[("573-A",1),("573-F",1)], q={}, qd=0.25),
 "S7 franchise per tier":   dict(sel=[("573-A",1),("573-A",2),("573-F",1)], q={1:0.5}, qd=0.25),
 "S9 q0,5 hanya t1 573-A":  dict(sel=[("573-A",1),("573-A",2),("573-F",1)], q={}, qd=0.25,
                                 qcell={("573-A",1):0.5}),
 "S8 baseline, b t3 dasar": dict(sel=[("573-A",1),("573-A",2),("573-A",3),("573-F",1)], q={}, qd=0.25,
                                 b_dasar={("573-A",3)}),
}

ring, detail = [], []
TH = np.array(sorted(set(TH_OF[TH_OF >= 2002])))
for nama, cfg in SKENARIO.items():
    qd, qmap, bdas = cfg["qd"], cfg.get("q", {}), cfg.get("b_dasar", set())
    qcell = cfg.get("qcell", {})
    agg = np.zeros(len(TH)); prem_tot = 0.0; n_tot = 0.0
    rows = []
    for (z, k) in cfg["sel"]:
        q = qcell.get((z,k), qmap.get(k, qd)); p = P_OPT[(z,k)]
        b = bk_row(z, k); bval = float(b.b_base if (z,k) in bdas else b.b_pakai)
        P, u = payout(z, k, q, p)
        he, pbayar = he_p10(z, k, q, p)
        n = eksposur(z, k)
        prem_kapal = P.sum()/len(TH)*bval if False else P.mean()*4*bval
        per_th = np.array([P[TH_OF == y].sum() for y in TH])
        agg += n*bval*per_th
        prem_kapal = per_th.mean()*bval
        prem_tot += n*prem_kapal; n_tot += n
        rows.append(dict(skenario=nama, zona=z, tier=k, q=q, p=p, u=u, b_k=bval, HE_p10=he,
                         p_bayar=pbayar, hari_thn=per_th.mean(), premi_thn=prem_kapal,
                         pct_R=100*prem_kapal/b.R_med, eksposur=n))
    D = pd.DataFrame(rows); detail.append(D)
    bs = agg[rng.integers(0, len(agg), size=(20000, len(agg)))].mean(axis=1) if False else None
    draw = rng.choice(agg, size=200000, replace=True)
    var995 = np.quantile(draw, 0.995); tvar = draw[draw >= var995].mean()
    mono = all(bool(np.all(np.diff(g.sort_values("tier").premi_thn.to_numpy()) > 0))
               for _, g in D.groupby("zona") if len(g) > 1)
    ring.append(dict(skenario=nama, n_sel=len(D), eksposur=round(n_tot), cakupan_pct=100*n_tot/71181,
        HE_p10_min=D.HE_p10.min(), HE_p10_bobot=np.average(D.HE_p10, weights=D.eksposur),
        pct_R_t1=float(D[(D.zona=="573-A")&(D.tier==1)].pct_R.iloc[0]) if ((D.zona=="573-A")&(D.tier==1)).any() else np.nan,
        premi_t1_A=float(D[(D.zona=="573-A")&(D.tier==1)].premi_thn.iloc[0]) if ((D.zona=="573-A")&(D.tier==1)).any() else np.nan,
        pct_R_maks=D.pct_R.max(), p_bayar_bobot=np.average(D.p_bayar, weights=D.eksposur),
        volume_premi_M=prem_tot/1e9, cv_agregat=agg.std(ddof=1)/agg.mean(),
        thn_terburuk_x=agg.max()/agg.mean(), tvar995_per_premi_INDIKATIF=agg.max()/agg.mean(), 
        modal_per_premi=(tvar-agg.mean())/agg.mean(), relativitas_monoton=mono))

R = pd.DataFrame(ring); R.to_csv(AKAR/"output/tables/simulasi_skenario.csv", index=False)
pd.concat(detail).to_csv(AKAR/"output/tables/simulasi_skenario_sel.csv", index=False)
pd.set_option("display.width", 260)
print(R.round(3).to_string(index=False))
v = pd.concat(detail); v = v[(v.skenario=="S1 baseline 20 Sep")]
print("\nvalidasi hari_thn S1 vs optimasi_final:")
F = pd.read_csv(AKAR/"output/tables/optimasi_final.csv")
for _, r in v.iterrows():
    ref = F[(F.zona==r.zona)&(F.tier==r.tier)].hari_bayar_thn.iloc[0]
    print(f"  {r.zona} t{r.tier}: sim {r.hari_thn:.3f} vs 17 {ref:.3f}  selisih {r.hari_thn-ref:+.3f}")
