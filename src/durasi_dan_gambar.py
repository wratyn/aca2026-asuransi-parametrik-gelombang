"""early duration fit + diagnostic figs (ERA5 2021-2024)"""
from pathlib import Path
import glob, json
import numpy as np, pandas as pd, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
import statsmodels.api as sm

AKAR = Path(__file__).resolve().parent.parent
RAW = str(AKAR / "data" / "raw")
OUT = str(AKAR / "output")
for _sub in ("tables", "figures"):
    (Path(OUT) / _sub).mkdir(parents=True, exist_ok=True)
T = 365.25
rng = np.random.default_rng(7)


def muat():
    fs = sorted(glob.glob(f"{RAW}/era5_swh_wpp573_20*.nc"))
    import os
    _a, _b = int(os.environ.get("ACA_TAHUN_AWAL", 2021)), int(os.environ.get("ACA_TAHUN_AKHIR", 2024))  # default 2021-2024
    fs = [f for f in fs if _a <= int(f[-7:-3]) <= _b]
    return xr.concat([xr.open_dataset(f) for f in fs], dim="valid_time")["swh"].load(
        ).rename({"valid_time": "waktu"})


def klasifikasi(da):
    laut = da.notnull().all("waktu").values
    darat = ~laut
    nlat, nlon = laut.shape
    pes = np.zeros_like(laut)
    for i in range(nlat):
        for j in range(nlon):
            if laut[i, j] and (darat[max(0, i-1):i+2, max(0, j-1):j+2].any()
                               or i in (0, nlat-1) or j in (0, nlon-1)):
                pes[i, j] = True
    return laut, pes


def matriks(da, mask):
    v = da.values; idx = np.argwhere(mask)
    return np.stack([v[:, i, j] for i, j in idx], axis=1), idx


def episode(x, u, jeda=2):
    over = np.asarray(x) > u
    if not over.any():
        return np.array([], int), np.array([], int)
    idx = np.flatnonzero(over)
    g = np.split(idx, np.flatnonzero(np.diff(idx) > jeda) + 1)
    return np.array([a[0] for a in g]), np.array([len(a) for a in g])


def pmf_geom(d, p):
    return stats.geom.pmf(d, p)


def pmf_ztnb(d, r, p):
    return stats.nbinom.pmf(d, r, p) / (1 - stats.nbinom.pmf(0, r, p))


def pmf_dweib(d, q, b):
    return q ** ((d - 1) ** b) - q ** (d ** b)


def fit(d, pmf, x0, bounds):
    nll = lambda th: -np.sum(np.log(np.maximum(pmf(d, *th), 1e-300)))
    r = optimize.minimize(nll, x0, bounds=bounds, method="L-BFGS-B")
    return r.x, -r.fun


def ks_bootstrap(d, pmf, th, sampler, B=500):
    def cdf(x, th):
        m = int(max(np.max(d) * 5, 200))
        s = np.cumsum(pmf(np.arange(1, m + 1), *th))
        return np.clip(np.interp(x, np.arange(1, m + 1), s), 0, 1)
    stat = stats.kstest(d, lambda x: cdf(x, th)).statistic
    cnt = 0
    for _ in range(B):
        ds = sampler(th, len(d))
        try:
            ths, _ = fit(ds, pmf, th, [(1e-4, 1 - 1e-4)] * len(th) if len(th) == 1
                         else [(1e-3, 50), (1e-4, 1 - 1e-4)])
        except Exception:
            ths = th
        if stats.kstest(ds, lambda x: cdf(x, ths)).statistic >= stat:
            cnt += 1
    return stat, (cnt + 1) / (B + 1)


def main():
    da = muat()
    waktu = pd.to_datetime(da.waktu.values)
    n_thn = len({t.year for t in waktu})
    laut, pes = klasifikasi(da)
    mat_pes, idx_pes = matriks(da, pes)
    lonv = da.longitude.values
    lon_pes = np.array([lonv[j] for _, j in idx_pes])
    zona = np.digitize(lon_pes, np.arange(104, 129, 4.0))
    bid = pd.Series(waktu).dt.to_period("M").astype(str).values

    hasil = {}
    dur_all, seri_z1 = [], None
    for z in np.unique(zona):
        kol = np.flatnonzero(zona == z)
        if len(kol) < 3:
            continue
        s = np.nanmean(mat_pes[:, kol], axis=1)
        u = float(np.nanpercentile(s, 90))
        _, d = episode(s, u, jeda=2)
        dur_all.append(d)
        if z == 1:
            seri_z1, u_z1 = s, u
    d = np.concatenate(dur_all)
    hasil["durasi"] = dict(n=int(len(d)), mean=float(d.mean()),
                           var=float(d.var(ddof=1)), maks=int(d.max()),
                           rasio_var_mean=float(d.var(ddof=1) / d.mean()))

    cand = {}
    th, ll = fit(d, pmf_geom, [0.3], [(1e-4, 1 - 1e-4)])
    st, p = ks_bootstrap(d, pmf_geom, th, lambda t, n: stats.geom.rvs(t[0], size=n, random_state=rng))
    cand["Geometrik"] = dict(par=[round(float(x), 4) for x in th], loglik=round(ll, 2),
                             aic=round(-2*ll + 2*1, 2), ks=round(float(st), 4), ks_p=round(p, 4))

    th2, ll2 = fit(d, pmf_ztnb, [2.0, 0.4], [(1e-3, 50), (1e-4, 1 - 1e-4)])
    def samp_ztnb(t, n):
        out = []
        while len(out) < n:
            x = stats.nbinom.rvs(t[0], t[1], size=n, random_state=rng)
            out += list(x[x > 0])
        return np.array(out[:n])
    st2, p2 = ks_bootstrap(d, pmf_ztnb, th2, samp_ztnb)
    cand["NB terpotong-nol"] = dict(par=[round(float(x), 4) for x in th2], loglik=round(ll2, 2),
                                    aic=round(-2*ll2 + 2*2, 2), ks=round(float(st2), 4), ks_p=round(p2, 4))

    th3, ll3 = fit(d, pmf_dweib, [0.5, 1.0], [(1e-4, 1 - 1e-4), (0.1, 5)])
    cand["Weibull diskret"] = dict(par=[round(float(x), 4) for x in th3], loglik=round(ll3, 2),
                                   aic=round(-2*ll3 + 2*2, 2))
    hasil["fit_durasi"] = cand

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    hari = (mat_pes > 2.0).sum(axis=0) / n_thn
    ax[0].scatter(lon_pes, hari, s=14, c=zona, cmap="viridis", alpha=.8)
    ax[0].set_xlabel("Bujur (°E)"); ax[0].set_ylabel("Hari SWH > 2 m per tahun")
    ax[0].set_title("(a) Heterogenitas iklim gelombang\ndi sel pesisir WPP 573")
    ax[0].grid(alpha=.3)

    kk = np.arange(1, d.max() + 1)
    ax[1].bar(kk, [np.mean(d == k) for k in kk], color="#9ecae1", label="empiris")
    ax[1].plot(kk, pmf_geom(kk, th[0]), "o--", color="#d95f02", ms=4, label="geometrik")
    ax[1].plot(kk, pmf_ztnb(kk, *th2), "s-", color="#1b7837", ms=4, label="NB terpotong-nol")
    ax[1].set_xlabel("Durasi episode (hari)"); ax[1].set_ylabel("Proporsi")
    ax[1].set_title("(b) Distribusi durasi episode\n(ambang relatif p90, semua zona)")
    ax[1].legend(); ax[1].grid(alpha=.3)

    mulai, _ = episode(seri_z1, u_z1, jeda=2)
    ep_b = pd.Series(0.0, index=pd.unique(bid))
    vc = pd.Series(bid[mulai]).value_counts(); ep_b.loc[vc.index] = vc.values.astype(float)
    df = pd.DataFrame({"w": waktu, "b": bid})
    t_mid = df.groupby("b")["w"].apply(lambda s: (s.mean() - waktu.min()).days
                                       ).reindex(pd.unique(bid)).values.astype(float)
    pjg = df.groupby("b").size().reindex(pd.unique(bid)).values.astype(float)
    X = np.column_stack([np.ones_like(t_mid)] +
                        [f(2*np.pi*j*t_mid/T) for j in (1,) for f in (np.cos, np.sin)])
    m = sm.GLM(ep_b.values, X, family=sm.families.Poisson(), offset=np.log(pjg)).fit()
    tg = np.linspace(0, T, 400)
    Xg = np.column_stack([np.ones_like(tg)] +
                         [f(2*np.pi*j*tg/T) for j in (1,) for f in (np.cos, np.sin)])
    lam = np.exp(Xg @ m.params)
    bulan_num = pd.Series(waktu[mulai]).dt.month
    rata = np.array([ (bulan_num == b).sum() / n_thn / 30.4 for b in range(1, 13) ])
    ax[2].bar(np.arange(1, 13), rata, color="#c6dbef", label="empiris (per hari)")
    ax[2].plot(tg / 30.4 + 0.5, lam, color="#08519c", lw=2, label="λ(t) Fourier J=1")
    ax[2].set_xlabel("Bulan"); ax[2].set_ylabel("Intensitas episode (per hari)")
    ax[2].set_title("(c) Intensitas siklis λ(t), zona 1\n(pesisir selatan Jawa barat)")
    ax[2].set_xticks(range(1, 13)); ax[2].legend(); ax[2].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/fig_basisrisk_diagnostik.png", dpi=150)
    hasil["glm_zona1_J1"] = dict(params=[round(float(x), 4) for x in m.params],
                                 aic=round(float(m.aic), 2))

    tz = pd.read_csv(f"{OUT}/tables/tab_adu_rancangan_per_zona.csv")
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.2))
    for r, c, lbl in [("A", "#d73027", "A: absolut + indeks WPP"),
                      ("B", "#fdae61", "B: absolut + indeks zona"),
                      ("C", "#1a9850", "C: relatif p90 + indeks zona")]:
        s = tz[tz.rancangan == r]
        ax2[0].plot(s.zona, s.he_med, "o-", color=c, label=lbl)
        ax2[1].plot(s.zona, s.ce_gain_med / 1e3, "o-", color=c, label=lbl)
    ax2[0].axhline(0, color="k", lw=.8); ax2[1].axhline(0, color="k", lw=.8)
    ax2[0].set_xlabel("Zona (barat → timur)"); ax2[0].set_ylabel("Hedging effectiveness")
    ax2[0].set_title("(a) HE per zona"); ax2[0].legend(fontsize=8); ax2[0].grid(alpha=.3)
    ax2[1].set_xlabel("Zona (barat → timur)"); ax2[1].set_ylabel("Kenaikan CE (ribu Rp/musim)")
    ax2[1].set_title("(b) Manfaat kesejahteraan"); ax2[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/figures/fig_adu_rancangan.png", dpi=150)

    with open(f"{OUT}/tables/hasil_durasi.json", "w") as f:
        json.dump(hasil, f, indent=2, default=float)
    print(json.dumps(hasil, indent=2, default=float))


if __name__ == "__main__":
    main()
