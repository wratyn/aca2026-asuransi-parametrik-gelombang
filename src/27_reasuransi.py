#!/usr/bin/env python3
"""reinsurance structures"""
from pathlib import Path
import numpy as np, pandas as pd
from statistics import NormalDist
_ND = NormalDist()
class norm:
    cdf = staticmethod(np.vectorize(_ND.cdf)); ppf = staticmethod(np.vectorize(_ND.inv_cdf))
AKAR = Path(__file__).resolve().parent.parent
M_K, L, KMAKS = 7, 20, 400
MUSIM = ['Barat','Peralihan I','Timur','Peralihan II']
SEL = [("573-A",1), ("573-A",2), ("573-F",1)]
SKEN = {"A ketat": dict(coc=0.10, biaya=0.20), "B ramah": dict(coc=0.06, biaya=0.15)}
LAMS = [0.20, 0.30, 0.45]; LAM0 = 0.30
COC_R, K_DIV = 0.08, 0.35
rng = np.random.default_rng(20260921)

PM = pd.read_csv(AKAR/"data/processed/pmf_durasi.csv")
AB = pd.read_csv(AKAR/"output/tables/panjer_ab0.csv")
BK = pd.read_csv(AKAR/"output/tables/b_k_usulan.csv")
E  = pd.read_csv(AKAR/"data/processed/eksposur.csv")
NS = pd.read_csv(AKAR/"data/dim/dim_zona.csv").set_index("zona").n_sel
HT = pd.read_csv(AKAR/"data/processed/hari_terpicu_musim.csv")
eks = E.groupby(["zona","tier"]).n_kapal_zona.sum()
porsi_F = NS["573-F"]/(NS["573-D"]+NS["573-F"])
def eksposur(z, k):
    return float(eks.get(("573-B", k), 0.0)) if z == "573-A" else float(eks.get(("573-E", k), 0.0))*porsi_F
def bk(z, k):
    r = BK[(BK.zona == ("573-A" if z == "573-A" else "573-DF")) & (BK.tier == k)]
    return float(r.b_pakai.iloc[0])
def panjer_ab0(a, b, p0, fD, K=KMAKS):
    g = np.zeros(K+1); g[0] = p0
    for x in range(1, K+1):
        g[x] = sum((a + b*y/x)*fD[y]*g[x-y] for y in range(1, min(x, len(fD)-1)+1))
    return g
def binom_ab0(lam, phi):
    p = min(max(1-phi, 1e-6), 0.999); m = max(int(round(lam/p)), 1); p = lam/m
    return -p/(1-p), (m+1)*p/(1-p), (1-p)**m

dist, fdict = {}, {}
for (z, k) in SEL:
    pm = PM[(PM.zona == z) & (PM.tier == k)].sort_values("d")
    fD = np.zeros(M_K+1); fD[1:] = pm.prob_empiris.to_numpy(); fD[1:] /= fD[1:].sum()
    for s in MUSIM:
        r = AB[(AB.zona==z)&(AB.tier==k)&(AB.musim==s)&(AB.cacah=="Binomial")].iloc[0]
        a, b, p0 = binom_ab0(float(r.lambda_musim), float(r.phi_dipakai))
        g = panjer_ab0(a, b, p0, fD); x = np.arange(len(g))
        dist[(z,k,s)] = dict(cdf=np.cumsum(g), payout=np.minimum(np.clip(x-int(r.f), 0, None), L))
        fdict[(z,k,s)] = int(r.f)

N = 100_000
agg = np.zeros(N); per_sel = {}
for z in sorted({z for z,_ in SEL}):
    for s in MUSIM:
        U = rng.random(N)
        for (zz, k) in [c for c in SEL if c[0] == z]:
            d = dist[(zz,k,s)]
            W = np.searchsorted(d["cdf"], U)
            bay = d["payout"][np.clip(W, 0, len(d["payout"])-1)]*bk(zz,k)
            agg += eksposur(zz,k)*bay
            per_sel[(zz,k)] = per_sel.get((zz,k), np.zeros(N)) + bay
E_agg = agg.mean()
tot_sel = {c: eksposur(*c)*per_sel[c] for c in SEL}
share = {c: np.divide(tot_sel[c], agg, out=np.zeros(N), where=agg > 0) for c in SEL}

order = np.argsort(agg, kind="stable")
def wang_w(lam):
    S = 1 - np.arange(N+1)/N
    gS = norm.cdf(norm.ppf(np.clip(S, 1e-12, 1-1e-12)) + lam); gS[0], gS[-1] = 1.0, 0.0
    w = np.empty(N); w[order] = gS[:-1] - gS[1:]
    return w
W = {lam: wang_w(lam) for lam in LAMS}
def tvar_cap(x, q=0.995):
    v = np.quantile(x, q); return x[x >= v].mean() - x.mean(), x >= v
def VaR(T): return float(np.quantile(agg, 1-1/T)) if np.isfinite(T) else np.inf

def evaluasi(label, ret, ced, gov, harga="wang", lam=LAM0):
    cap_ret, ekor = tvar_cap(ret)
    Ec, Eg = ced.mean(), gov.mean()
    if ced.any():
        P_R = float((W[lam]*ced).sum()) if harga == "wang" else Ec + COC_R*K_DIV*tvar_cap(ced)[0]
    else: P_R = 0.0
    out = dict(struktur=label, harga=harga, lam=lam if harga=="wang" else np.nan,
               E_ret=ret.mean(), modal_ret=cap_ret, E_ced=Ec, P_RI=P_R, loading_RI=P_R-Ec,
               multiple_RI=P_R/Ec if Ec > 0 else np.nan, E_gov=Eg,
               p_sentuh_RI=float((ced > 0).mean()), p_sentuh_gov=float((gov > 0).mean()))
    for n, c in SKEN.items():
        margin = c["coc"]*cap_ret + (P_R - Ec)
        out[f"margin_{n}"] = margin
        out[f"bruto_{n}"] = (E_agg + margin)/(1-c["biaya"])
    sel = {}
    for cc in SEL:
        rc, cd = ret*share[cc], ced*share[cc]
        m_ret = rc[ekor].mean() - rc.mean()
        if ced.any():
            l_ri = float((W[lam]*cd).sum()) - cd.mean() if harga == "wang" else \
                   (P_R - Ec)*cd.mean()/Ec
        else: l_ri = 0.0
        sel[cc] = (m_ret, l_ri)
    return out, sel

base = {}
rows, selrows = [], []
def catat(o, s):
    rows.append(o)
    for cc, (m_ret, l_ri) in s.items():
        n = eksposur(*cc); murni = per_sel[cc].mean()
        r = dict(struktur=o["struktur"], harga=o["harga"], lam=o["lam"], zona=cc[0], tier=cc[1],
                 premi_murni=murni)
        for nm, c in SKEN.items():
            r[f"bruto_{nm}"] = (murni + (c["coc"]*m_ret + l_ri)/n)/(1-c["biaya"])
        selrows.append(r)

zero = np.zeros(N)
catat(*evaluasi("S0 tanpa reasuransi", agg, zero, zero))
for harga, lams in [("wang", LAMS), ("coc_div", [np.nan])]:
    for lam in lams:
        catat(*evaluasi("S1 quota share 50%", 0.5*agg, 0.5*agg, zero, harga, lam))
        for Ta in [1.5, 2, 3, 4, 5, 7, 10, 15, 20]:
            for Tx in [50, 100, 200, 500, np.inf]:
                a, x = VaR(Ta), VaR(Tx)
                ced = np.clip(agg, a, x) - a
                catat(*evaluasi(f"S2 XoL agregat {Ta:g}->{Tx:g} thn", agg-ced, ced, zero, harga, lam))
        for Ta in [2, 3, 5, 10]:
            for TxR in [20, 50, 100]:
                a, x = VaR(Ta), VaR(TxR)
                ced = np.clip(agg, a, x) - a; gov = np.clip(agg - x, 0, None)
                catat(*evaluasi(f"S3 XoL {Ta:g}->{TxR:g} + pemerintah >{TxR:g} thn",
                                agg-ced-gov, ced, gov, harga, lam))
G = pd.DataFrame(rows); S = pd.DataFrame(selrows)
b0 = G.iloc[0]
for n in SKEN: G[f"d_bruto_pct_{n}"] = 100*(G[f"bruto_{n}"]/b0[f"bruto_{n}"] - 1)
G.to_csv(AKAR/"output/tables/reas_grid.csv", index=False)
S.to_csv(AKAR/"output/tables/reas_per_sel.csv", index=False)

H = HT[HT.apply(lambda r: (r.zona, r.tier) in SEL, axis=1)].copy()
H["f"] = [fdict[(r.zona, r.tier, r.musim)] for r in H.itertuples()]
H["bayar"] = [min(max(r.W - r.f, 0), L)*bk(r.zona, r.tier)*eksposur(r.zona, r.tier) for r in H.itertuples()]
Hy = H.groupby("th_musim").bayar.sum()
Hy = Hy[H.groupby("th_musim").musim.nunique() == 4]
Hy.rename("agregat").to_frame().assign(
    periode_ulang_model=lambda d: [1/max((agg >= v).mean(), 1/N) for v in d.agregat]
).to_csv(AKAR/"output/tables/reas_historis.csv")

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
kol = ["struktur","harga","lam","modal_ret","E_ced","P_RI","multiple_RI","p_sentuh_RI",
       "bruto_A ketat","d_bruto_pct_A ketat","d_bruto_pct_B ramah"]
print(f"E_agg {E_agg/1e9:.1f} M | VaR: " + ", ".join(f"{T:g}th {VaR(T)/1e9:.0f}M" for T in [1.5,2,3,5,10,20,50,100,200,500]))
print("tahun historis:", len(Hy), "| rata2 M", round(Hy.mean()/1e9,1), "| maks M", round(Hy.max()/1e9,1))
print((Hy/1e9).round(0).to_string())
g = G[kol].copy(); 
for c in ["modal_ret","E_ced","P_RI","bruto_A ketat"]: g[c] = g[c]/1e9
print(g[(g.harga=="wang")&(g.lam==LAM0) | (g.struktur.str.startswith("S0"))].round(2).to_string(index=False))
