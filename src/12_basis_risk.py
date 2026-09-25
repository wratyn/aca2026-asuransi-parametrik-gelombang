#!/usr/bin/env python3
"""NTN as loss proxy (didn't work)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.formula.api as smf
AKAR = Path(__file__).resolve().parent.parent

def rakit(cb, ntn, bridge, tier):
    b = bridge[bridge.zona != "-"].copy()
    b["w"] = b.groupby("provinsi").bobot.transform(lambda s: s/s.sum())
    c = cb[cb.tier == tier][["zona","tahun","bulan","n_hari_terpicu","n_episode","musim"]]
    m = b.merge(c, on="zona")
    m["hp"] = m.n_hari_terpicu * m.w
    m["ep"] = m.n_episode * m.w
    agg = (m.groupby(["provinsi","tahun","bulan"])
             .agg(hari_terpicu=("hp","sum"), n_episode=("ep","sum"), musim=("musim","first"))
             .reset_index())
    d = ntn.merge(agg, on=["provinsi","tahun","bulan"], how="inner")
    return d

def he(d, formula_penuh, formula_tanpa):
    f = smf.ols(formula_penuh, data=d).fit()
    t = smf.ols(formula_tanpa, data=d).fit()
    return f, t, 1 - f.resid.var(ddof=0)/t.resid.var(ddof=0)

def main():
    cb  = pd.read_csv(AKAR/"data/processed/cacah_bulanan.csv")
    ntn = pd.read_csv(AKAR/"data/processed/panel_ntn.csv")
    br  = pd.read_csv(AKAR/"data/dim/bridge_provinsi_zona.csv")
    ntn["tangkap_murni"] = ntn.sumber.str.contains("tangkap") | ntn.sumber.str.startswith("NTN")

    F_PENUH = "ntn ~ C(provinsi) + C(musim) + hari_terpicu"
    F_TANPA = "ntn ~ C(provinsi) + C(musim)"

    print("="*84)
    print("A. REGRESI UTAMA — per tier, ambang awal (persentil 85/90/95)")
    print("="*84)
    print(f"{'tier':<5} {'sampel':>7} {'n_prov':>7} {'gamma':>9} {'se':>7} {'p':>9} {'HE':>7} {'R2 penuh':>9}")
    print("-"*64)
    baris=[]
    for tier in (1,2,3):
        d = rakit(cb, ntn, br, tier)
        f,t,h = he(d, F_PENUH, F_TANPA)
        g = f.params["hari_terpicu"]; se = f.bse["hari_terpicu"]; p = f.pvalues["hari_terpicu"]
        print(f"{tier:<5} {len(d):>7} {d.provinsi.nunique():>7} {g:>9.4f} {se:>7.4f} {p:>9.2e} {h:>7.4f} {f.rsquared:>9.4f}")
        baris.append(dict(sampel="semua provinsi", tier=tier, n=len(d), gamma=g, se=se, p=p, HE=h, r2=f.rsquared))

    print()
    print("="*84)
    print("B. ROBUSTNESS — hanya provinsi dengan NTN Perikanan TANGKAP murni")
    print("="*84)
    print(f"{'tier':<5} {'sampel':>7} {'n_prov':>7} {'gamma':>9} {'se':>7} {'p':>9} {'HE':>7} {'R2 penuh':>9}")
    print("-"*64)
    for tier in (1,2,3):
        d = rakit(cb, ntn[ntn.tangkap_murni], br, tier)
        f,t,h = he(d, F_PENUH, F_TANPA)
        g = f.params["hari_terpicu"]; se = f.bse["hari_terpicu"]; p = f.pvalues["hari_terpicu"]
        print(f"{tier:<5} {len(d):>7} {d.provinsi.nunique():>7} {g:>9.4f} {se:>7.4f} {p:>9.2e} {h:>7.4f} {f.rsquared:>9.4f}")
        baris.append(dict(sampel="tangkap murni", tier=tier, n=len(d), gamma=g, se=se, p=p, HE=h, r2=f.rsquared))
    pd.DataFrame(baris).to_csv(AKAR/"output/tables/basis_risk_regresi.csv", index=False)

    print()
    print("="*84)
    print("C. PER ZONA — provinsi dikelompokkan menurut zona dominannya (tier 1)")
    print("="*84)
    b2 = br[br.zona!="-"].copy()
    dom = b2.loc[b2.groupby("provinsi").bobot.idxmax()][["provinsi","zona"]]
    d = rakit(cb, ntn, br, 1).merge(dom, on="provinsi")
    print(f"{'zona':<8} {'n_prov':>7} {'sampel':>7} {'gamma':>9} {'p':>9} {'HE':>7}")
    print("-"*52)
    for z, g_ in d.groupby("zona"):
        if g_.provinsi.nunique() < 2:
            f = smf.ols("ntn ~ C(musim) + hari_terpicu", data=g_).fit()
            t = smf.ols("ntn ~ C(musim)", data=g_).fit()
        else:
            f = smf.ols(F_PENUH, data=g_).fit(); t = smf.ols(F_TANPA, data=g_).fit()
        h = 1 - f.resid.var(ddof=0)/t.resid.var(ddof=0)
        print(f"{z:<8} {g_.provinsi.nunique():>7} {len(g_):>7} {f.params['hari_terpicu']:>9.4f} "
              f"{f.pvalues['hari_terpicu']:>9.2e} {h:>7.4f}")

if __name__ == "__main__":
    main()
