#!/usr/bin/env python3
"""impact, final scenario (24 Sep)"""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR/"src"))
import zonasi as Z
spec = importlib.util.spec_from_file_location("pipa", AKAR/"src/10_pipeline_indeks.py")
pipa = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipa)

L, M_K = 20, 7
MUSIM = ['Barat','Peralihan I','Timur','Peralihan II']
SEL = [("573-A",1,85), ("573-A",2,90), ("573-F",1,86)]
PK  = {1:85, 2:90}
ZNUM = {"573-A":1, "573-D":4, "573-F":6}

PR = pd.read_csv(AKAR/"output/tables/premi.csv")
RS = pd.read_csv(AKAR/"output/tables/reas_per_sel.csv")
RS = RS[(RS.struktur=="S3 XoL 5->20 + pemerintah >20 thn")&(RS.harga=="wang")&(RS.lam.round(2)==0.30)]
BIAYA_BARU = 0.12
def premi_final(z, k):
    r = RS[(RS.zona==z)&(RS.tier==k)].iloc[0]
    A, B, pm = r["bruto_A ketat"], r["bruto_B ramah"], r.premi_murni
    K = (0.80*A - 0.85*B)/0.04
    Lri = 0.80*A - pm - 0.10*K
    return (pm + 0.10*K + Lri)/(1-BIAYA_BARU)
PR["bruto_Lama A 20%"] = PR["bruto_A ketat"]
PR["bruto_Final 12%+RI"] = [premi_final(z,k) for z,k in zip(PR.zona, PR.tier)]
AB = pd.read_csv(AKAR/"output/tables/panjer_ab0.csv")
BK = pd.read_csv(AKAR/"output/tables/b_k_usulan.csv")
def bkrow(z,k):
    r = BK[(BK.zona==("573-A" if z=="573-A" else "573-DF"))&(BK.tier==k)]; return r.iloc[0]

h = Z.jalankan(simpan=False, verbose=False)
mat, label, lat, lon = h["mat"], h["label"], h["lat"], h["lon"]
wkt = pd.to_datetime(h["waktu"])
mus = np.array([pipa.MUSIM[m] for m in wkt.month])
thm = np.where(wkt.month == 12, wkt.year+1, wkt.year)
KOL = [(s,y) for s in MUSIM for y in range(2001,2026) if not (s=='Barat' and y==2001)]
KOLI = {k:i for i,k in enumerate(KOL)}
S_OF = np.array([MUSIM.index(s) for s,_ in KOL]); TH = np.array([y for _,y in KOL])

def W_musim(x, u):
    out = np.zeros(len(KOL))
    m, a = pipa.episode(x, u)
    for s0, s1 in zip(m, a):
        kk = (mus[s0], thm[s0])
        if kk in KOLI: out[KOLI[kk]] += min(s1-s0+1, M_K)
    return out

TAHUN = np.array(sorted(set(TH[TH >= 2002])))
def per_tahun(v): return np.array([v[TH == y].sum() for y in TAHUN])

rows, ring = [], []
for (z, k, p) in SEL:
    zn = ZNUM[z]; M = mat[:, label == zn]; idx = M.mean(axis=1)
    b = float(bkrow(z,k).b_pakai); R = float(bkrow(z,k).R_med)
    pr = PR[(PR.zona==z)&(PR.tier==k)].iloc[0]
    u_z = np.percentile(idx, p)
    Wz = W_musim(idx, u_z)
    f = np.zeros(len(Wz))
    for s in range(4):
        fs = int(AB[(AB.zona==z)&(AB.tier==k)&(AB.musim==MUSIM[s])&(AB.cacah=="Binomial")].f.iloc[0])
        f[S_OF == s] = fs
    P = np.minimum(np.clip(Wz - f, 0, None), L)
    P_th = per_tahun(P)
    for sken, prem in (("Lama A 20%", pr["bruto_Lama A 20%"]), ("Final 12%+RI", pr["bruto_Final 12%+RI"])):
        for sub in (0.0, 0.5, 0.7, 0.8):
            bayar_sendiri = prem*(1-sub)
            for c in range(M.shape[1]):
                Xi = W_musim(M[:, c], np.percentile(M[:, c], PK[k]))
                X_th = per_tahun(Xi)
                tanpa = b*X_th
                dengan = b*X_th + bayar_sendiri - b*P_th
                tim = (S_OF == 2)
                X_tim = np.array([Xi[tim][i] for i in range(tim.sum())])
                P_tim = P[tim]
                rows.append(dict(zona=z, tier=k, lat=lat[label==zn][c], lon=lon[label==zn][c],
                    skenario=sken, subsidi=sub, premi_dibayar=bayar_sendiri,
                    rata_tanpa=tanpa.mean(), rata_dengan=dengan.mean(),
                    sd_tanpa=tanpa.std(ddof=1), sd_dengan=dengan.std(ddof=1),
                    p90_tanpa=np.percentile(tanpa,90), p90_dengan=np.percentile(dengan,90),
                    maks_tanpa=tanpa.max(), maks_dengan=dengan.max(),
                    rasio_tutup=(b*P_th).mean()/max((b*X_th).mean(),1),
                    tutup_tahun_buruk=(b*P_th[np.argmax(X_th)])/max(b*X_th.max(),1),
                    hari_hilang_timur=X_tim.mean(), hari_dibayar_timur=P_tim.mean()))
D = pd.DataFrame(rows); D.to_csv(AKAR/"output/tables/dampak_per_sel_24sep.csv", index=False)

for (z,k,sken,sub), g in D.groupby(["zona","tier","skenario","subsidi"]):
    ring.append(dict(zona=z, tier=k, skenario=sken, subsidi=sub, n_sel=len(g),
        premi_dibayar=g.premi_dibayar.iloc[0],
        d_sd_pct=100*(g.sd_dengan.mean()/g.sd_tanpa.mean()-1),
        d_p90_pct=100*(g.p90_dengan.mean()/g.p90_tanpa.mean()-1),
        d_maks_pct=100*(g.maks_dengan.mean()/g.maks_tanpa.mean()-1),
        d_rata_pct=100*(g.rata_dengan.mean()/g.rata_tanpa.mean()-1),
        sel_p90_membaik=100*(g.p90_dengan < g.p90_tanpa).mean(),
        sel_maks_membaik=100*(g.maks_dengan < g.maks_tanpa).mean(),
        rasio_tutup=g.rasio_tutup.mean(), tutup_tahun_buruk=g.tutup_tahun_buruk.mean(),
        hari_hilang_timur=g.hari_hilang_timur.mean(), hari_dibayar_timur=g.hari_dibayar_timur.mean()))
Rg = pd.DataFrame(ring); Rg.to_csv(AKAR/"output/tables/dampak_ringkas_24sep.csv", index=False)

sub_rows = []
for sken in ("Lama A 20%","Final 12%+RI"):
    for sub in (0.0, 0.5, 0.7, 0.8):
        tot = 0.0; n = 0
        for (z,k,_) in SEL:
            pr = PR[(PR.zona==z)&(PR.tier==k)].iloc[0]
            tot += pr.eksposur*pr[f"bruto_{sken}"]*sub; n += pr.eksposur
        sub_rows.append(dict(skenario=sken, subsidi=sub, kapal=n, anggaran_subsidi_M=tot/1e9))
SB = pd.DataFrame(sub_rows); SB.to_csv(AKAR/"output/tables/subsidi_24sep.csv", index=False)

pd.set_option("display.width", 280)
print("== dampak per sel pesisir (rata-rata antar sel) ==")
print(Rg.round(2).to_string(index=False))
print("\n== anggaran subsidi ==")
print(SB.round(2).to_string(index=False))
