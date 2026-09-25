"""sanity checks on key numbers"""
import glob
import numpy as np, pandas as pd, xarray as xr
from scipy import stats

from pathlib import Path as _Path
AKAR = _Path(__file__).resolve().parent.parent
RAW = str(AKAR / "data" / "raw")
ok = lambda b: "OK " if b else "GAGAL "

def episode(x, u, jeda=2):
    over = np.asarray(x) > u
    if not over.any():
        return np.array([], int), np.array([], int)
    idx = np.flatnonzero(over)
    g = np.split(idx, np.flatnonzero(np.diff(idx) > jeda) + 1)
    return np.array([a[0] for a in g]), np.array([len(a) for a in g])

x = np.array([0,3,3,0,3,0,0,0,0,3,0,3,3,3,0])
m, d = episode(x, 1, jeda=2)
print(ok(list(m) == [1, 9] and list(d) == [3, 4]),
      f"dekluster: mulai={list(m)} durasi={list(d)} (harap [1,9] / [3,4])")

L = np.array([0., 1, 2, 3, 4, 5, 6, 7])
for a in (1.0, 1.5, 2.0, 0.5):
    I = a * L
    he = 1 - (L - I).var() / L.var()
    print(ok(abs(he - (1 - (a - 1) ** 2)) < 1e-12),
          f"HE untuk I={a}L -> {he:.4f} (teori {1-(a-1)**2:.4f})")

lam, M = 3.7, 5
N = stats.poisson.rvs(lam, size=400_000, random_state=np.random.default_rng(1))
emp = np.minimum(N, M).mean()
teori = sum(1 - stats.poisson.cdf(xx, lam) for xx in range(M))
print(ok(abs(emp - teori) < 5e-3), f"E[min(N,M)]: simulasi {emp:.4f} vs identitas {teori:.4f}")

q = 0.3454
def panjer(lam, fD, K):
    g = np.zeros(K + 1); g[0] = np.exp(-lam)
    for xx in range(1, K + 1):
        s = sum(y * fD[y] * g[xx - y] for y in range(1, min(xx, len(fD) - 1) + 1))
        g[xx] = lam / xx * s
    return g
K = 400
fD = np.zeros(K + 1); fD[1:] = stats.geom.pmf(np.arange(1, K + 1), q)
g = panjer(3.7, fD, K)
rg = np.random.default_rng(2)
Ns = rg.poisson(3.7, 300_000)
W = np.array([stats.geom.rvs(q, size=n, random_state=rg).sum() if n else 0 for n in Ns[:60_000]])
emp_mean, teo_mean = W.mean(), 3.7 * (1 / q)
print(ok(abs(emp_mean - teo_mean) < 0.15), f"E[W]: simulasi {emp_mean:.3f} vs Wald {teo_mean:.3f}")
print(ok(abs(g.sum() - 1) < 1e-6), f"Panjer: total massa {g.sum():.8f}")
emp_p0 = (W == 0).mean()
print(ok(abs(emp_p0 - g[0]) < 5e-3), f"P(W=0): simulasi {emp_p0:.4f} vs Panjer {g[0]:.4f}")

fs = sorted(glob.glob(f"{RAW}/era5_swh_wpp573_20*.nc"))
import os
_a, _b = int(os.environ.get("ACA_TAHUN_AWAL", 2021)), int(os.environ.get("ACA_TAHUN_AKHIR", 2024))  # default 2021-2024
fs = [f for f in fs if _a <= int(f[-7:-3]) <= _b]
ds = xr.concat([xr.open_dataset(f) for f in fs], dim="valid_time")
swh = ds["swh"]
laut = swh.notnull().all("valid_time")
hari = (swh > 2.0).sum("valid_time") / 4.0
h = hari.where(laut).values
h = h[~np.isnan(h)]
print(f"    [cek] seluruh sel laut: min={h.min():.1f} median={np.median(h):.1f} "
      f"maks={h.max():.1f} hari/tahun di atas 2 m")
idxmax = np.unravel_index(np.nanargmax(np.where(laut.values, hari.values, np.nan)), hari.shape)
print(f"    [cek] sel terkasar di {swh.latitude.values[idxmax[0]]:.1f}N "
      f"{swh.longitude.values[idxmax[1]]:.1f}E")
print(f"    [cek] SWH maks tercatat {float(swh.max()):.2f} m, "
      f"rata-rata seluruh domain {float(swh.mean()):.2f} m")

v = swh.values
mask = laut.values
pes = np.zeros_like(mask)
nlat, nlon = mask.shape
for i in range(nlat):
    for j in range(nlon):
        if mask[i, j] and ((~mask)[max(0,i-1):i+2, max(0,j-1):j+2].any()
                           or i in (0, nlat-1) or j in (0, nlon-1)):
            pes[i, j] = True
kol = np.argwhere(pes)
lonv = swh.longitude.values
sel_z1 = [(i, j) for i, j in kol if lonv[j] < 108]
s1 = np.nanmean(np.stack([v[:, i, j] for i, j in sel_z1], 1), 1)
u = np.percentile(s1, 90)
t = pd.to_datetime(swh.valid_time.values)
bid = pd.Series(t).dt.to_period("M").astype(str).values
hari_b = pd.Series((s1 > u).astype(float)).groupby(bid).sum()
m2, d2 = episode(s1, u, 2)
epb = pd.Series(0.0, index=pd.unique(bid))
vc = pd.Series(bid[m2]).value_counts(); epb.loc[vc.index] = vc.values.astype(float)
print(f"    [cek] zona1 u=p90={u:.2f}: var/mean HARI={hari_b.var()/hari_b.mean():.2f}, "
      f"EPISODE={epb.var()/epb.mean():.2f}, n_episode={len(d2)}")
print(ok(hari_b.var()/hari_b.mean() > 2 and abs(epb.var()/epb.mean() - 1) < 0.8),
      "overdispersi hari >> episode, sesuai klaim dekluster")
