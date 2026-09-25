#!/usr/bin/env python3
"""(a,b,0) count choice -> binomial, reprice"""
from pathlib import Path
import numpy as np, pandas as pd
AKAR = Path(__file__).resolve().parent.parent
M_K, L, Q, KMAKS = 7, 20, 0.25, 400
MUSIM = ['Barat','Peralihan I','Timur','Peralihan II']
DIJUAL = {("573-A",1), ("573-A",2), ("573-F",1)}

EP = pd.read_csv(AKAR/"data/processed/episode.csv")
PM = pd.read_csv(AKAR/"data/processed/pmf_durasi.csv")
DP = pd.read_csv(AKAR/"output/tables/dist_panjer.csv")
BK = pd.read_csv(AKAR/"output/tables/b_k_usulan.csv")
def b_of(z, k):
    r = BK[(BK.zona == ("573-A" if z == "573-A" else "573-DF")) & (BK.tier == k)]
    return float(r.b_pakai.iloc[0]) if len(r) else np.nan

def panjer_ab0(a, b, p0, fD, K=KMAKS):
    g = np.zeros(K+1); g[0] = p0
    for x in range(1, K+1):
        s = 0.0
        for y in range(1, min(x, len(fD)-1)+1):
            s += (a + b*y/x)*fD[y]*g[x-y]
        g[x] = s
    return g

def poisson_ab0(lam): return 0.0, lam, np.exp(-lam)
def binom_ab0(lam, phi):
    p = min(max(1-phi, 1e-6), 0.999)
    m = max(int(round(lam/p)), 1)
    p = lam/m
    return -p/(1-p), (m+1)*p/(1-p), (1-p)**m

def kuantil(g, q): return int(np.searchsorted(np.cumsum(g), q))

EP["W"] = np.minimum(EP.durasi, M_K)
disp, hasil = [], []
for (z, k), e in EP.groupby(["zona","tier"]):
    tahun = sorted(EP.tahun.unique())
    for s in MUSIM:
        es = e[e.musim == s]
        n_th = np.array([ (es.tahun == y).sum() for y in tahun ], float)
        if s == 'Barat': n_th = n_th[1:]
        W_th = np.array([ es[es.tahun == y].W.sum() for y in tahun ], float)
        if s == 'Barat': W_th = W_th[1:]
        mean_n, var_n = n_th.mean(), n_th.var(ddof=1)
        disp.append(dict(zona=z, tier=k, musim=s, n_tahun=len(n_th), rata_cacah=mean_n,
                         var_cacah=var_n, phi=var_n/mean_n if mean_n > 0 else np.nan,
                         E_W_data=W_th.mean(), sd_W_data=W_th.std(ddof=1)))
D = pd.DataFrame(disp); D.to_csv(AKAR/"output/tables/dispersi_cacah.csv", index=False)

for _, r in D.iterrows():
    z, k, s = r.zona, int(r.tier), r.musim
    lam = float(DP[(DP.zona == z) & (DP.tier == k) & (DP.musim == s)].lambda_musim.iloc[0])
    pm = PM[(PM.zona == z) & (PM.tier == k)].sort_values("d")
    fD = np.zeros(M_K+1); fD[1:] = pm.prob_empiris.to_numpy(); fD[1:] /= fD[1:].sum()
    phi = float(np.clip(r.phi, 0.05, 1.0))
    for nama, (a, b, p0) in (("Poisson", poisson_ab0(lam)), ("Binomial", binom_ab0(lam, phi))):
        g = panjer_ab0(a, b, p0, fD); x = np.arange(len(g))
        EW = float((x*g).sum()); sdW = float(np.sqrt((x**2*g).sum() - EW**2))
        f = kuantil(g, Q); P = np.minimum(np.clip(x - f, 0, None), L)
        hasil.append(dict(zona=z, tier=k, musim=s, cacah=nama, phi_dipakai=phi if nama=="Binomial" else 1.0,
                          lambda_musim=lam, massa=float(g.sum()), E_W=EW, sd_W=sdW,
                          E_W_data=r.E_W_data, sd_W_data=r.sd_W_data, f=f,
                          E_payout=float((P*g).sum()), p_bayar=float(g[x > f].sum()),
                          W_q99=kuantil(g, 0.99), dijual=(z,k) in DIJUAL))
H = pd.DataFrame(hasil); H["premi_musim"] = H.E_payout*[b_of(z,k) for z,k in zip(H.zona, H.tier)]
H.to_csv(AKAR/"output/tables/panjer_ab0.csv", index=False)

pd.set_option("display.width", 260)
print("== dispersi cacah episode per musim (phi = var/mean) ==")
print(D[D.set_index(["zona","tier"]).index.isin(DIJUAL)].round(3).to_string(index=False))
print("\n== massa Panjer ==", H.massa.min().round(10), H.massa.max().round(10))
print("\n== sel S2: model vs data ==")
v = H[H.dijual]
print(v[["zona","tier","musim","cacah","phi_dipakai","E_W","E_W_data","sd_W","sd_W_data","f","E_payout","p_bayar"]]
      .round(2).to_string(index=False))
print("\n== premi murni per tahun ==")
t = v.groupby(["zona","tier","cacah"]).agg(hari_thn=("E_payout","sum"), premi_thn=("premi_musim","sum")).reset_index()
print(t.round(2).to_string(index=False))
