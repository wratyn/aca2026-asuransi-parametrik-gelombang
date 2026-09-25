"""franchise candidates per zone x tier x season"""
import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
M_K, L = 7, 20
e = pd.read_csv(ROOT/'data/processed/episode.csv')
e['bln'] = pd.to_datetime(e.tgl_mulai).dt.month
e['th_musim'] = np.where(e.bln == 12, e.tahun + 1, e.tahun)
e['d'] = e.durasi.clip(upper=M_K)
W = e.groupby(['zona','tier','musim','th_musim']).d.sum()
idx = pd.MultiIndex.from_product([sorted(e.zona.unique()), [1,2,3], ['Barat','Peralihan I','Timur','Peralihan II'], range(2001, 2026)],
                                 names=W.index.names)
W = W.reindex(idx, fill_value=0).rename('W').reset_index()
W = W[~((W.musim == 'Barat') & (W.th_musim == 2001))]
W.to_csv(ROOT/'data/processed/hari_terpicu_musim.csv', index=False)

bk = pd.read_csv(ROOT/'output/tables/b_k_usulan.csv')
def b_of(z, k):
    zz = '573-A' if z == '573-A' else '573-DF'
    r = bk[(bk.zona == zz) & (bk.tier == k)]
    return float(r.b_pakai.iloc[0]) if len(r) else np.nan

ATURAN = {'f0_capW20': None, 'f_min': 'min', 'f_q25': 0.25, 'f_median': 0.5}
rows = []
for (z, k, s), g in W.groupby(['zona','tier','musim']):
    w = g.W.to_numpy()
    for nama, a in ATURAN.items():
        if a is None:
            f, bayar = 0, np.minimum(w, L)
        else:
            f = int(w.min()) if a == 'min' else int(np.floor(np.quantile(w, a)))
            bayar = np.minimum(np.clip(w - f, 0, None), L)
        mu = bayar.mean(); b = b_of(z, k)
        rows.append(dict(zona=z, tier=k, musim=s, aturan=nama, f=f, n_musim=len(w),
            W_min=w.min(), W_med=np.median(w), W_mean=w.mean(), W_max=w.max(),
            p_bayar=(bayar > 0).mean(), hari_bayar_mean=mu,
            cv=bayar.std(ddof=1)/mu if mu > 0 else np.nan,
            porsi_pasti=bayar.min()/mu if mu > 0 else np.nan,
            porsi_cap=(bayar >= L).mean(),
            cakupan=mu/w.mean() if w.mean() > 0 else np.nan,
            b_k=b, premi_murni=mu*b))
T = pd.DataFrame(rows)
T.to_csv(ROOT/'output/tables/franchise_kandidat.csv', index=False)

A = T.groupby(['zona','tier','aturan']).agg(premi_murni_tahun=('premi_murni','sum'), hari_bayar_tahun=('hari_bayar_mean','sum')).reset_index()
bk2 = bk.set_index(['zona','tier']).R_med
A['R_k'] = [bk2.get(('573-A' if z=='573-A' else '573-DF', k), np.nan) for z, k in zip(A.zona, A.tier)]
A['premi_pct_R'] = 100*A.premi_murni_tahun/A.R_k
A.to_csv(ROOT/'output/tables/franchise_ringkas_tahunan.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(T[['zona','tier','musim','aturan','f','W_min','W_med','W_max','p_bayar','hari_bayar_mean','cv','porsi_pasti','porsi_cap','cakupan','premi_murni']].round(2).to_string())
print(A.round(2).to_string())
