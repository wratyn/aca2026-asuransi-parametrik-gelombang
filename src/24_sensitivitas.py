#!/usr/bin/env python3
"""one-at-a-time sensitivity"""
from pathlib import Path
import numpy as np, pandas as pd
AKAR = Path(__file__).resolve().parent.parent
M_K, L0, Q0, KMAKS = 7, 20, 0.25, 400
MUSIM = ['Barat','Peralihan I','Timur','Peralihan II']
SEL = [("573-A",1), ("573-A",2), ("573-F",1)]
COC0, BIAYA0 = 0.10, 0.20
rng = np.random.default_rng(20260921)

PM = pd.read_csv(AKAR/"data/processed/pmf_durasi.csv")
AB = pd.read_csv(AKAR/"output/tables/panjer_ab0.csv")
BK = pd.read_csv(AKAR/"output/tables/b_k_usulan.csv")
E  = pd.read_csv(AKAR/"data/processed/eksposur.csv")
NS = pd.read_csv(AKAR/"data/dim/dim_zona.csv").set_index("zona").n_sel
G  = pd.read_csv(AKAR/"output/tables/optimasi_grid.csv")
eks = E.groupby(["zona","tier"]).n_kapal_zona.sum()
porsi_F = NS["573-F"]/(NS["573-D"]+NS["573-F"])
def eksposur(z,k):
    return float(eks.get(("573-B",k),0.0)) if z=="573-A" else float(eks.get(("573-E",k),0.0))*porsi_F
def bkrow(z,k):
    r = BK[(BK.zona==("573-A" if z=="573-A" else "573-DF"))&(BK.tier==k)]; return r.iloc[0]

def panjer_ab0(a,b,p0,fD,K=KMAKS):
    g = np.zeros(K+1); g[0] = p0
    for x in range(1,K+1):
        g[x] = sum((a+b*y/x)*fD[y]*g[x-y] for y in range(1,min(x,len(fD)-1)+1))
    return g
def par_binom(lam,phi):
    p = min(max(1-phi,1e-6),0.999); m = max(int(round(lam/p)),1); p = lam/m
    return -p/(1-p),(m+1)*p/(1-p),(1-p)**m
def par_poisson(lam): return 0.0, lam, np.exp(-lam)

def jalankan(q=Q0, L=L0, phi_x=1.0, lam_x=1.0, sev="empiris", cacah="Binomial", b_x=1.0,
             NTHN=100_000):
    D, premi = {}, {}
    for (z, k) in SEL:
        pm = PM[(PM.zona==z)&(PM.tier==k)].sort_values("d")
        kol = "prob_empiris" if sev=="empiris" else "prob"
        fD = np.zeros(M_K+1); fD[1:] = pm[kol].to_numpy(); fD[1:] /= fD[1:].sum()
        b = float(bkrow(z,k).b_pakai)*b_x
        tot = 0.0
        for s in MUSIM:
            r = AB[(AB.zona==z)&(AB.tier==k)&(AB.musim==s)&(AB.cacah=="Binomial")].iloc[0]
            lam = float(r.lambda_musim)*lam_x
            phi = float(np.clip(r.phi_dipakai*phi_x, 0.05, 1.0))
            a_, b_, p0 = par_binom(lam, phi) if cacah=="Binomial" else par_poisson(lam)
            g = panjer_ab0(a_, b_, p0, fD); x = np.arange(len(g)); cdf = np.cumsum(g)
            f = int(np.searchsorted(cdf, q)) if q > 0 else 0
            P = np.minimum(np.clip(x-f, 0, None), L)
            D[(z,k,s)] = dict(cdf=cdf, bayar=P*b)
            tot += float((P*g).sum())*b
        premi[(z,k)] = tot
    agg = np.zeros(NTHN)
    for z in sorted({zz for zz,_ in SEL}):
        for s in MUSIM:
            U = rng.random(NTHN)
            for (zz, k) in [c for c in SEL if c[0]==z]:
                d = D[(zz,k,s)]
                W = np.clip(np.searchsorted(d["cdf"], U), 0, len(d["bayar"])-1)
                agg += eksposur(zz,k)*d["bayar"][W]
    v = np.quantile(agg, 0.995); modal = agg[agg>=v].mean() - agg.mean()
    return premi, agg.mean(), modal

def bruto(premi_murni, modal, E_agg, coc=COC0, biaya=BIAYA0):
    faktor = coc*modal/E_agg
    return {c: (p*(1+faktor))/(1-biaya) for c,p in premi_murni.items()}

dasar, Eag0, modal0 = jalankan()
br0 = bruto(dasar, modal0, Eag0)
KUNCI = ("573-A",1)
baris = [dict(faktor="kasus dasar", nilai="q=0,25; L=20; binomial; b bersih; CoC 10%; biaya 20%",
              premi_murni_t1=dasar[KUNCI], bruto_t1=br0[KUNCI], d_bruto_pct=0.0,
              modal_per_premi=modal0/Eag0, d_modal_pct=0.0)]

SKEN = [
 ("franchise q", "q = 0 (tanpa franchise)",        dict(q=0.0)),
 ("franchise q", "q = 0,5 (median)",               dict(q=0.5)),
 ("cap L",       "L = 15 hari",                    dict(L=15)),
 ("cap L",       "L = 25 hari",                    dict(L=25)),
 ("dispersi",    "phi x 0,8 (lebih teratur)",      dict(phi_x=0.8)),
 ("dispersi",    "phi x 1,25",                     dict(phi_x=1.25)),
 ("dispersi",    "Poisson (phi = 1)",              dict(cacah="Poisson")),
 ("intensitas",  "lambda +10% (iklim memburuk)",   dict(lam_x=1.10)),
 ("intensitas",  "lambda -10%",                    dict(lam_x=0.90)),
 ("severitas",   "Weibull diskret (parametrik)",   dict(sev="parametrik")),
 ("manfaat b_k", "b_k -25% (batas bawah literatur)", dict(b_x=0.75)),
 ("manfaat b_k", "b_k +67% (batas atas literatur)",  dict(b_x=1.67)),
]
for fak, label, kw in SKEN:
    pm_, Eag, modal = jalankan(**kw)
    br = bruto(pm_, modal, Eag)
    baris.append(dict(faktor=fak, nilai=label, premi_murni_t1=pm_[KUNCI], bruto_t1=br[KUNCI],
                      d_bruto_pct=100*(br[KUNCI]/br0[KUNCI]-1),
                      modal_per_premi=modal/Eag, d_modal_pct=100*((modal/Eag)/(modal0/Eag0)-1)))
for coc, biaya, label in ((0.06,0.15,"CoC 6%, biaya 15%"), (0.15,0.25,"CoC 15%, biaya 25%")):
    br = bruto(dasar, modal0, Eag0, coc, biaya)
    baris.append(dict(faktor="loading", nilai=label, premi_murni_t1=dasar[KUNCI], bruto_t1=br[KUNCI],
                      d_bruto_pct=100*(br[KUNCI]/br0[KUNCI]-1), modal_per_premi=modal0/Eag0, d_modal_pct=0.0))

S = pd.DataFrame(baris); S["bruto_pct_R"] = 100*S.bruto_t1/float(bkrow(*KUNCI).R_med)
S.to_csv(AKAR/"output/tables/sensitivitas.csv", index=False)

rows = []
for (z,k) in SEL:
    g = G[(G.zona==z)&(G.tier==k)&(G.q==0.25)].sort_values("p")
    p_opt = {("573-A",1):85, ("573-A",2):90, ("573-F",1):86}[(z,k)]
    ref = g[g.p==p_opt].iloc[0]
    for dp in (-5,-2,0,2,5):
        r = g[g.p==p_opt+dp]
        if len(r)==0: continue
        r = r.iloc[0]
        rows.append(dict(zona=z, tier=k, p=int(r.p), d_p=dp, u=r.u_z, HE_p10=r.HE_p10,
                         premi_rel=r.premi_murni_thn/ref.premi_murni_thn,
                         p_bayar=r.p_bayar, d_HE=r.HE_p10-ref.HE_p10))
SP = pd.DataFrame(rows); SP.to_csv(AKAR/"output/tables/sensitivitas_p.csv", index=False)

pd.set_option("display.width", 260)
print("== sensitivitas (premi bruto 573-A tier 1) ==")
print(S.round(3).to_string(index=False))
print("\n== sensitivitas ambang p (jalur empiris, q = 0,25) ==")
print(SP.round(3).to_string(index=False))
