#!/usr/bin/env python3
"""gross premium, capital (TVaR 99.5), RBC"""
from pathlib import Path
import numpy as np, pandas as pd
AKAR = Path(__file__).resolve().parent.parent
M_K, L, Q, KMAKS = 7, 20, 0.25, 400
MUSIM = ['Barat','Peralihan I','Timur','Peralihan II']
SEL = [("573-A",1), ("573-A",2), ("573-F",1)]
SKEN = {"A ketat": dict(coc=0.10, biaya=0.20), "B ramah": dict(coc=0.06, biaya=0.15)}
rng = np.random.default_rng(20260921)

PM = pd.read_csv(AKAR/"data/processed/pmf_durasi.csv")
AB = pd.read_csv(AKAR/"output/tables/panjer_ab0.csv")
BK = pd.read_csv(AKAR/"output/tables/b_k_usulan.csv")
E  = pd.read_csv(AKAR/"data/processed/eksposur.csv")
NS = pd.read_csv(AKAR/"data/dim/dim_zona.csv").set_index("zona").n_sel
eks = E.groupby(["zona","tier"]).n_kapal_zona.sum()
porsi_F = NS["573-F"]/(NS["573-D"]+NS["573-F"])
def eksposur(z, k):
    if z == "573-A": return float(eks.get(("573-B", k), 0.0))
    return float(eks.get(("573-E", k), 0.0))*porsi_F
def bk(z, k):
    r = BK[(BK.zona == ("573-A" if z == "573-A" else "573-DF")) & (BK.tier == k)]
    return float(r.b_pakai.iloc[0]), float(r.R_med.iloc[0])

def panjer_ab0(a, b, p0, fD, K=KMAKS):
    g = np.zeros(K+1); g[0] = p0
    for x in range(1, K+1):
        g[x] = sum((a + b*y/x)*fD[y]*g[x-y] for y in range(1, min(x, len(fD)-1)+1))
    return g
def binom_ab0(lam, phi):
    p = min(max(1-phi, 1e-6), 0.999); m = max(int(round(lam/p)), 1); p = lam/m
    return -p/(1-p), (m+1)*p/(1-p), (1-p)**m

dist = {}
for (z, k) in SEL:
    pm = PM[(PM.zona == z) & (PM.tier == k)].sort_values("d")
    fD = np.zeros(M_K+1); fD[1:] = pm.prob_empiris.to_numpy(); fD[1:] /= fD[1:].sum()
    for s in MUSIM:
        r = AB[(AB.zona==z)&(AB.tier==k)&(AB.musim==s)&(AB.cacah=="Binomial")].iloc[0]
        a, b, p0 = binom_ab0(float(r.lambda_musim), float(r.phi_dipakai))
        g = panjer_ab0(a, b, p0, fD); x = np.arange(len(g))
        P = np.minimum(np.clip(x - int(r.f), 0, None), L)
        dist[(z,k,s)] = dict(g=g, cdf=np.cumsum(g), payout=P, f=int(r.f),
                             E=float((P*g).sum()), p_bayar=float(g[x > int(r.f)].sum()))

NTHN = 100_000
agg = np.zeros(NTHN); per_sel = {}
for z in sorted({z for z,_ in SEL}):
    for s in MUSIM:
        U = rng.random(NTHN)
        for (zz, k) in [c for c in SEL if c[0] == z]:
            d = dist[(zz,k,s)]
            W = np.searchsorted(d["cdf"], U)
            bay = d["payout"][np.clip(W, 0, len(d["payout"])-1)]*bk(zz,k)[0]
            agg += eksposur(zz,k)*bay
            per_sel[(zz,k)] = per_sel.get((zz,k), np.zeros(NTHN)) + bay
E_agg = agg.mean()
var995 = np.quantile(agg, 0.995); tvar995 = agg[agg >= var995].mean()
modal = tvar995 - E_agg

rows = []
for (z, k) in SEL:
    b, R = bk(z, k); n = eksposur(z, k)
    murni = per_sel[(z,k)].mean()
    ekor = per_sel[(z,k)][agg >= var995].mean()
    rows.append(dict(zona=z, tier=k, eksposur=n, b_k=b, R_k=R, premi_murni=murni,
                     premi_murni_total=n*murni, sd_polis=per_sel[(z,k)].std(),
                     tvar_polis=ekor, modal_euler=n*(ekor-murni)))
P = pd.DataFrame(rows)
P["pangsa_modal"] = P.modal_euler/P.modal_euler.sum()
for nama, cfg in SKEN.items():
    margin_tot = cfg["coc"]*modal
    P[f"margin_{nama}"] = margin_tot*P.pangsa_modal/P.eksposur
    P[f"bruto_{nama}"] = (P.premi_murni + P[f"margin_{nama}"])/(1-cfg["biaya"])
    P[f"pctR_{nama}"] = 100*P[f"bruto_{nama}"]/P.R_k
    P[f"bulan_{nama}"] = P[f"bruto_{nama}"]/12
P.to_csv(AKAR/"output/tables/premi.csv", index=False)

base = P[(P.zona=="573-A")&(P.tier==1)].iloc[0]
R = P[["zona","tier","premi_murni"]].copy()
for nama in SKEN: R[f"rel_{nama}"] = P[f"bruto_{nama}"]/base[f"bruto_{nama}"]
R["rel_murni"] = P.premi_murni/base.premi_murni
R.to_csv(AKAR/"output/tables/relativitas.csv", index=False)

M = pd.DataFrame([dict(E_agregat=E_agg, sd_agregat=agg.std(), cv=agg.std()/E_agg,
    VaR995=var995, TVaR995=tvar995, modal=modal, modal_per_premi=modal/E_agg,
    thn_terburuk_x=agg.max()/E_agg, p_nol=float((agg == 0).mean()),
    **{f"margin_{n}": SKEN[n]["coc"]*modal for n in SKEN},
    **{f"premi_bruto_total_{n}": float((P[f"bruto_{n}"]*P.eksposur).sum()) for n in SKEN})])
M.to_csv(AKAR/"output/tables/modal_rbc.csv", index=False)

pd.set_option("display.width", 280)
print("== premi per kapal per tahun ==")
print(P[["zona","tier","eksposur","premi_murni","margin_A ketat","bruto_A ketat","pctR_A ketat",
         "bruto_B ramah","pctR_B ramah"]].round(0).to_string(index=False))
print("\n== portofolio (100.000 tahun simulasi, komonoton dalam zona) ==")
print(M.round(3).T.to_string())
print("\n== relativitas terhadap 573-A tier 1 ==")
print(R.round(3).to_string(index=False))
