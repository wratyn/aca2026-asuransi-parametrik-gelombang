#!/usr/bin/env python3
"""cyclic Poisson GLM (Fourier) per zone x tier"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
from scipy import stats
AKAR = Path(__file__).resolve().parent.parent
T = 365.25

def rancang(g, J, tren=True):
    X = [np.ones(len(g))]
    for j in range(1, J+1):
        X += [np.cos(2*np.pi*j*g.t_absolut/T), np.sin(2*np.pi*j*g.t_absolut/T)]
    if tren: X.append(g.t_absolut.values/365.25)
    return np.column_stack(X)

def fit(g, J):
    X = rancang(g, J)
    return sm.GLM(g.n_episode, X, family=sm.families.Poisson(),
                  offset=np.log(g.hari)).fit()

def main():
    cb = pd.read_csv(AKAR/"data/processed/cacah_bulanan.csv")
    ep = pd.read_csv(AKAR/"data/processed/episode.csv")
    hasil = []
    print(f"{'zona':<8} {'tier':<5} {'J':>2} {'AIC':>9} {'phi':>6} {'p_siklis':>10} {'p_KS':>7} {'lam/thn':>8} {'tren %/thn':>11}")
    print("-"*78)
    for (z,t), g in cb.groupby(["zona","tier"]):
        g = g.sort_values(["tahun","bulan"])
        m0 = sm.GLM(g.n_episode, np.column_stack([np.ones(len(g)), g.t_absolut/365.25]),
                    family=sm.families.Poisson(), offset=np.log(g.hari)).fit()
        kand = {J: fit(g, J) for J in (1,2,3)}
        J = min(kand, key=lambda j: kand[j].aic)
        m = kand[J]
        mu = m.fittedvalues
        phi = float(((g.n_episode - mu)**2 / mu).sum() / (len(g) - m.df_model - 1))
        lr = 2*(m.llf - m0.llf); dof = 2*J
        p_sik = float(stats.chi2.sf(lr, dof))
        e = ep[(ep.zona==z)&(ep.tier==t)].sort_values("tgl_mulai")
        tt = (pd.to_datetime(e.tgl_mulai) - pd.to_datetime(cb.iloc[0].tahun*10000+101, format="%Y%m%d")).dt.days.values
        b = np.asarray(m.params)
        def lam(s):
            v = b[0]
            for j in range(1, J+1):
                v += b[2*j-1]*np.cos(2*np.pi*j*s/T) + b[2*j]*np.sin(2*np.pi*j*s/T)
            return np.exp(v + b[-1]*s/365.25)
        grid = np.arange(tt.min(), tt.max()+1)
        Lam = np.concatenate([[0], np.cumsum(lam(grid))])
        idx = np.clip(tt - tt.min(), 0, len(Lam)-1)
        dL = np.diff(Lam[idx]); dL = dL[dL > 0]
        p_ks = float(stats.kstest(dL, "expon").pvalue) if len(dL) > 10 else np.nan
        lam_thn = float(g.n_episode.sum()/ (g.hari.sum()/365.25))
        tren = (np.exp(b[-1])-1)*100
        print(f"{z:<8} {t:<5} {J:>2} {m.aic:>9.1f} {phi:>6.2f} {p_sik:>10.2e} {p_ks:>7.3f} {lam_thn:>8.1f} {tren:>+10.2f}%")
        hasil.append(dict(zona=z, tier=t, J=J, a0=b[0], **{f"a{j}":b[2*j-1] for j in range(1,J+1)},
                          **{f"b{j}":b[2*j] for j in range(1,J+1)}, gamma=b[-1],
                          aic=m.aic, bic=m.bic, phi_dispersi=phi, lr_p_vs_J0=p_sik,
                          ks_p_rescaling=p_ks, n_episode_total=int(g.n_episode.sum()),
                          lambda_per_tahun=lam_thn, tren_persen_per_tahun=tren))
    pd.DataFrame(hasil).to_csv(AKAR/"output/tables/fit_lambda.csv", index=False)
    print("\n-> output/tables/fit_lambda.csv")
    print("syarat lolos: phi dalam [0,7 , 1,4] · p_siklis < 0,05 · p_KS > 0,05")

if __name__ == "__main__":
    main()
