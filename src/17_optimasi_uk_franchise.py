#!/usr/bin/env python3
"""grid search trigger percentile + franchise, max HE p10"""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR/"src"))
import zonasi as Z
spec = importlib.util.spec_from_file_location("pipa", AKAR/"src/10_pipeline_indeks.py")
pipa = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipa)

KUNCI = {1:"573-A", 4:"573-D", 6:"573-F"}
PK = {1:85, 2:90, 3:95}
GRID_P = list(range(70, 98))
GRID_Q = [0.0, 0.25, 0.5]
M_K, L = 7, 20
MUSIM = ['Barat','Peralihan I','Timur','Peralihan II']

h = Z.jalankan(simpan=False, verbose=False)
mat, label, lat, lon = h["mat"], h["label"], h["lat"], h["lon"]
wkt = pd.to_datetime(h["waktu"])
mus = np.array([pipa.MUSIM[m] for m in wkt.month])
thm = np.where(wkt.month == 12, wkt.year + 1, wkt.year)
KOL = [(s, y) for s in MUSIM for y in range(2001, 2026) if not (s == 'Barat' and y == 2001)]
KOLI = {k: i for i, k in enumerate(KOL)}
S_OF = np.array([MUSIM.index(s) for s, _ in KOL])

def W_musim(x, u):
    out = np.zeros(len(KOL))
    m, a = pipa.episode(x, u)
    for s0, s1 in zip(m, a):
        k = (mus[s0], thm[s0])
        if k in KOLI:
            out[KOLI[k]] += min(s1 - s0 + 1, M_K)
    return out

def f_musim(W, q):
    f = np.zeros(len(W))
    if q == 0: return f
    for s in range(4):
        sel = S_OF == s
        f[sel] = np.floor(np.quantile(W[sel], q))
    return f

def demean(v):
    out = v.copy().astype(float)
    for s in range(4):
        sel = S_OF == s; out[sel] -= v[sel].mean()
    return out

bk = pd.read_csv(AKAR/"output/tables/b_k_usulan.csv")
def b_of(z, k):
    r = bk[(bk.zona == ('573-A' if z == '573-A' else '573-DF')) & (bk.tier == k)]
    return float(r.b_pakai.iloc[0]) if len(r) else np.nan

baris, per_sel = [], []
for zn, z in KUNCI.items():
    M = mat[:, label == zn]; idx = M.mean(axis=1)
    la, lo = lat[label == zn], lon[label == zn]
    for k, Pk in PK.items():
        X = np.array([W_musim(M[:, c], np.percentile(M[:, c], Pk)) for c in range(M.shape[1])])
        Xd = np.array([demean(x) for x in X]); varX = (Xd**2).sum(axis=1)
        buruk = np.array([np.concatenate([x[S_OF == s] > np.median(x[S_OF == s]) for s in range(4)]) for x in X])
        urut = np.concatenate([np.where(S_OF == s)[0] for s in range(4)])
        B = np.zeros_like(buruk); B[:, urut] = buruk
        for p in GRID_P:
            Wz = W_musim(idx, np.percentile(idx, p))
            for q in GRID_Q:
                f = f_musim(Wz, q)
                P = np.minimum(np.clip(Wz - f, 0, None), L)
                rd = np.array([demean(x - P) for x in X])
                he = 1 - (rd**2).sum(axis=1)/varX
                fn = np.array([((P == 0) & B[c]).sum()/max(B[c].sum(), 1) for c in range(len(X))])
                hari_thn = sum(P[S_OF == s].mean() for s in range(4))
                baris.append(dict(zona=z, tier=k, P_rugi=Pk, p=p, q=q, u_z=round(float(np.percentile(idx, p)), 4),
                    HE_rata=he.mean(), HE_p10=np.percentile(he, 10), HE_min=he.min(),
                    FN_buruk_rata=fn.mean(), FN_buruk_p90=np.percentile(fn, 90),
                    p_bayar=(P > 0).mean(), hari_bayar_thn=hari_thn,
                    premi_murni_thn=hari_thn*b_of(z, k)))
                if (p == Pk) or q == 0.25:
                    for c in range(len(X)):
                        per_sel.append(dict(zona=z, tier=k, p=p, q=q, lat=la[c], lon=lo[c], HE=he[c], FN_buruk=fn[c]))

G = pd.DataFrame(baris); G.to_csv(AKAR/"output/tables/optimasi_grid.csv", index=False)
pd.DataFrame(per_sel).to_csv(AKAR/"data/processed/optimasi_per_sel.csv", index=False)

pil = []
for (z, k, q), g in G.groupby(['zona','tier','q']):
    b = g.loc[g.HE_p10.idxmax()]
    plat = g[g.HE_p10 >= b.HE_p10 - 0.02].p
    awal = g[g.p == PK[k]].iloc[0]
    pil.append(dict(zona=z, tier=k, q=q, p_opt=int(b.p), plateau=f"{plat.min()}-{plat.max()}", u_opt=b.u_z,
                    HE_p10_opt=b.HE_p10, HE_rata_opt=b.HE_rata, HE_min_opt=b.HE_min, FN_buruk_opt=b.FN_buruk_rata,
                    p_bayar_opt=b.p_bayar, premi_opt=b.premi_murni_thn,
                    HE_p10_awal=awal.HE_p10, premi_awal=awal.premi_murni_thn))
R = pd.DataFrame(pil); R.to_csv(AKAR/"output/tables/optimasi_ringkas.csv", index=False)
pd.set_option('display.width', 250)
print(R.round(3).to_string())

Q_FINAL = 0.25
fin, dimrows = [], []
for (z, k), g in G[G.q == Q_FINAL].groupby(['zona','tier']):
    g = g.sort_values('p').copy()
    g['HE_p10_halus'] = g.HE_p10.rolling(3, center=True, min_periods=2).mean()
    b = g.loc[g.HE_p10_halus.idxmax()]
    fin.append(dict(zona=z, tier=k, q=Q_FINAL, P_rugi=PK[k], p_opt=int(b.p), u_opt=b.u_z,
                    HE_p10=b.HE_p10, HE_p10_halus=b.HE_p10_halus, HE_rata=b.HE_rata, HE_min=b.HE_min,
                    FN_buruk=b.FN_buruk_rata, p_bayar=b.p_bayar, hari_bayar_thn=b.hari_bayar_thn,
                    premi_murni_thn=b.premi_murni_thn, layak=b.HE_p10_halus >= 0.40))
    zn = [n for n, v in KUNCI.items() if v == z][0]
    idx = mat[:, label == zn].mean(axis=1)
    Wz = W_musim(idx, b.u_z); f = f_musim(Wz, Q_FINAL)
    for s in range(4):
        sel = S_OF == s
        P = np.minimum(np.clip(Wz[sel] - f[sel], 0, None), L)
        dimrows.append(dict(zona=z, tier=k, musim=MUSIM[s], p_opt=int(b.p), u_k=b.u_z, q=Q_FINAL,
                            f_hari=int(f[sel][0]), L_hari=L, m_k=M_K, b_k=b_of(z, k),
                            W_rata=Wz[sel].mean(), hari_bayar_rata=P.mean(), p_bayar=(P > 0).mean(),
                            premi_murni_musim=P.mean()*b_of(z, k)))
F = pd.DataFrame(fin); F.to_csv(AKAR/"output/tables/optimasi_final.csv", index=False)
D = pd.DataFrame(dimrows); D.to_csv(AKAR/"data/dim/dim_desain_polis.csv", index=False)
print(F.round(3).to_string()); print(D.round(2).to_string())
