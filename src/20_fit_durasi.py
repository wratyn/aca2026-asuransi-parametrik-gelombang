#!/usr/bin/env python3
"""episode duration fit + Ferro-Segers theta"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize
AKAR = Path(__file__).resolve().parent.parent
M_K = 7
MUSIM = ['Barat','Peralihan I','Timur','Peralihan II']

ep = pd.read_csv(AKAR/"data/processed/episode.csv")
S  = pd.read_csv(AKAR/"data/processed/swh_harian_zona.csv", parse_dates=["tanggal"])

def nll_geom(par, d):
    p = 1/(1+np.exp(-par[0]))
    return -stats.geom.logpmf(d, p).sum()
def nll_nbtrunc(par, d):
    r, p = np.exp(par[0]), 1/(1+np.exp(-par[1]))
    lp = stats.nbinom.logpmf(d, r, p) - np.log1p(-stats.nbinom.pmf(0, r, p))
    return -lp.sum() if np.isfinite(lp).all() else 1e12
def sf_dw(d, q, beta):
    return q**(np.asarray(d, float)**beta)
def nll_dw(par, d):
    q, beta = 1/(1+np.exp(-par[0])), np.exp(par[1])
    pm = sf_dw(d-1, q, beta) - sf_dw(d, q, beta)
    return -np.log(np.clip(pm, 1e-300, None)).sum()

KEL = {"Geometrik": (nll_geom, [0.0], 1),
       "NB terpotong-nol": (nll_nbtrunc, [0.0, 0.0], 2),
       "Weibull diskret": (nll_dw, [0.0, 0.0], 2)}

def fit_semua(d):
    out = {}
    for nama, (f, x0, k) in KEL.items():
        r = optimize.minimize(f, x0, args=(d,), method="Nelder-Mead",
                              options=dict(maxiter=5000, xatol=1e-8, fatol=1e-8))
        out[nama] = dict(loglik=-r.fun, k=k, aic=2*r.fun+2*k, par=r.x)
    return out

def pmf_dw(par, dmax):
    q, beta = 1/(1+np.exp(-par[0])), np.exp(par[1])
    d = np.arange(1, dmax+1)
    return sf_dw(d-1, q, beta) - sf_dw(d, q, beta), sf_dw(dmax, q, beta)
def pmf_geom(par, dmax):
    p = 1/(1+np.exp(-par[0])); d = np.arange(1, dmax+1)
    return stats.geom.pmf(d, p), float(stats.geom.sf(dmax, p))
def pmf_nbt(par, dmax):
    r, p = np.exp(par[0]), 1/(1+np.exp(-par[1])); d = np.arange(1, dmax+1)
    pm = stats.nbinom.pmf(d, r, p)/(1-stats.nbinom.pmf(0, r, p))
    return pm, float(1-pm.sum())
PMF = {"Geometrik": pmf_geom, "NB terpotong-nol": pmf_nbt, "Weibull diskret": pmf_dw}

baris, pmfrows, theta = [], [], []
for (z, k), g in ep.groupby(["zona","tier"]):
    d = g.durasi.to_numpy()
    fits = fit_semua(d)
    terbaik = min(fits, key=lambda n: fits[n]["aic"])
    aic_musim = 0.0
    for s in MUSIM:
        ds = g[g.musim == s].durasi.to_numpy()
        if len(ds) >= 15:
            aic_musim += fit_semua(ds)[terbaik]["aic"]
        else:
            aic_musim = np.nan; break
    for nama, r in fits.items():
        baris.append(dict(zona=z, tier=k, distribusi=nama, n=len(d), loglik=r["loglik"],
                          k_par=r["k"], aic=r["aic"], terpilih=(nama == terbaik),
                          aic_per_musim=aic_musim if nama == terbaik else np.nan,
                          rerata_empiris=d.mean(), rerata_terpotong=np.minimum(d, M_K).mean()))
    pm, ekor = PMF[terbaik](fits[terbaik]["par"], M_K-1)
    pm_cap = np.append(pm, ekor)
    emp = np.array([(np.minimum(d, M_K) == x).mean() for x in range(1, M_K+1)])
    for i, x in enumerate(range(1, M_K+1)):
        pmfrows.append(dict(zona=z, tier=k, distribusi=terbaik, d=x, prob=pm_cap[i], prob_empiris=emp[i]))
    u = float(g.u_k.iloc[0]); x = S[S.zona == z].sort_values("tanggal").swh_rerata.to_numpy()
    idx = np.where(x > u)[0]
    T_ = np.diff(idx); N = len(T_)
    if N > 1:
        if T_.max() > 2:
            th = min(1.0, 2*(np.sum(T_-1)**2)/(N*np.sum((T_-1)*(T_-2))))
        else:
            th = min(1.0, 2*(T_.sum()**2)/(N*np.sum(T_**2)))
    else:
        th = np.nan
    theta.append(dict(zona=z, tier=k, u_k=u, n_lampau=len(idx), theta_FS=th,
                      gugus_rata_1_per_theta=1/th, durasi_rata_dekluster=d.mean(),
                      hari_lampau_per_episode=len(idx)/len(d)))

B = pd.DataFrame(baris); B.to_csv(AKAR/"output/tables/fit_durasi.csv", index=False)
P = pd.DataFrame(pmfrows); P.to_csv(AKAR/"data/processed/pmf_durasi.csv", index=False)
TH = pd.DataFrame(theta); TH.to_csv(AKAR/"output/tables/theta_ferro_segers.csv", index=False)
pd.set_option("display.width", 250)
print("== AIC per keluarga ==")
print(B.pivot_table(index=["zona","tier"], columns="distribusi", values="aic").round(1).to_string())
print("\n== terpilih, dan cek ketergantungan musim ==")
print(B[B.terpilih].drop(columns=["terpilih","k_par"]).round(2).to_string(index=False))
print("\n== theta Ferro-Segers ==")
print(TH.round(3).to_string(index=False))
print("\n== pmf terpotong di 7 hari (model vs empiris) ==")
print(P.pivot_table(index=["zona","tier"], columns="d", values=["prob","prob_empiris"]).round(3).to_string())
