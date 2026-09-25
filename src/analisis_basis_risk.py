"""first basis risk test: absolute vs relative trigger (ERA5 2021-2024)"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
from scipy import stats

AKAR = Path(__file__).resolve().parent.parent
RAW = str(AKAR / "data" / "raw")
OUT = str(AKAR / "output" / "tables")
Path(OUT).mkdir(parents=True, exist_ok=True)
T = 365.25
B_RUGI, B_BAYAR, CAP = 150_000.0, 100_000.0, 15


def muat():
    fs = sorted(glob.glob(f"{RAW}/era5_swh_wpp573_20*.nc"))
    import os
    _a, _b = int(os.environ.get("ACA_TAHUN_AWAL", 2021)), int(os.environ.get("ACA_TAHUN_AKHIR", 2024))  # default 2021-2024
    fs = [f for f in fs if _a <= int(f[-7:-3]) <= _b]
    ds = xr.concat([xr.open_dataset(f) for f in fs], dim="valid_time")
    return ds["swh"].load().rename({"valid_time": "waktu"})


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
    v = da.values
    idx = np.argwhere(mask)
    return np.stack([v[:, i, j] for i, j in idx], axis=1), idx


def episode(x, u, jeda=2):
    over = np.asarray(x) > u
    if not over.any():
        return np.array([], int), np.array([], int)
    idx = np.flatnonzero(over)
    grup = np.split(idx, np.flatnonzero(np.diff(idx) > jeda) + 1)
    return np.array([g[0] for g in grup]), np.array([len(g) for g in grup])


def main():
    da = muat()
    waktu = pd.to_datetime(da.waktu.values)
    n_thn = len({t.year for t in waktu})
    laut, pes = klasifikasi(da)
    mat_pes, idx_pes = matriks(da, pes)
    mat_laut, _ = matriks(da, laut)
    lonv = da.longitude.values
    lon_pes = np.array([lonv[j] for _, j in idx_pes])
    zona = np.digitize(lon_pes, np.arange(104, 129, 4.0))
    ix_wpp = np.nanmean(mat_laut, axis=1)

    musim = pd.Series(waktu).dt.month.map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}).values
    kunci = pd.Series(waktu).dt.year.astype(str).values + "-M" + pd.Series(musim).astype(str).values
    kidx = pd.Index(pd.unique(kunci))

    def agg(ind):
        return pd.DataFrame({"x": np.asarray(ind, float), "k": kunci}
                            ).groupby("k")["x"].sum().reindex(kidx).fillna(0.0)

    hasil = {}

    ringkas = []
    detail_zona = []
    for nama, mode in [("A. absolut 2,0 m + indeks WPP", "A"),
                       ("B. absolut 2,0 m + indeks zona", "B"),
                       ("C. relatif p90 lokal + indeks zona", "C")]:
        he_all, fn_all, prem_all, ce_all, bayar_all = [], [], [], [], []
        for z in np.unique(zona):
            kol = np.flatnonzero(zona == z)
            if len(kol) < 3:
                continue
            sub = mat_pes[:, kol]
            seri_zona = np.nanmean(sub, axis=1)
            if mode == "A":
                u_lokal = np.full(len(kol), 2.0)
                seri, u_idx = ix_wpp, 2.0
            elif mode == "B":
                u_lokal = np.full(len(kol), 2.0)
                seri, u_idx = seri_zona, 2.0
            else:
                u_lokal = np.nanpercentile(sub, 90, axis=0)
                seri, u_idx = seri_zona, float(np.nanpercentile(seri_zona, 90))

            I_hari = np.minimum(agg(seri > u_idx), CAP)
            I = I_hari * B_BAYAR
            prem = float(I.mean())
            he_z, fn_z, ce_z = [], [], []
            for m, k in enumerate(kol):
                Ld = sub[:, m] > u_lokal[m]
                L = agg(Ld) * B_RUGI
                if L.var() < 1e-6:
                    continue
                he_z.append(1 - (L - I).var() / L.var())
                rugi_hari = Ld.sum()
                fn_z.append(100 * (Ld & ~(seri > u_idx)).sum() / max(rugi_hari, 1))
                a, W = 2e-6, 5_000_000.0
                ce0 = -np.log(np.mean(np.exp(-a * (W - L)))) / a
                ce1 = -np.log(np.mean(np.exp(-a * (W - L + I - prem)))) / a
                ce_z.append(ce1 - ce0)
            if not he_z:
                continue
            detail_zona.append(dict(rancangan=mode, zona=int(z),
                                    lon=f"{lon_pes[kol].min():.0f}-{lon_pes[kol].max():.0f}E",
                                    u_indeks=round(float(u_idx), 2),
                                    u_lokal_med=round(float(np.median(u_lokal)), 2),
                                    hari_bayar_thn=float((seri > u_idx).sum() / n_thn),
                                    premi_musim=prem,
                                    he_med=float(np.median(he_z)),
                                    fn_med=float(np.median(fn_z)),
                                    ce_gain_med=float(np.median(ce_z))))
            he_all += he_z; fn_all += fn_z; ce_all += ce_z
            prem_all.append(prem); bayar_all.append((seri > u_idx).sum() / n_thn)
        ringkas.append(dict(rancangan=nama, n_sel=len(he_all),
                            hari_bayar_thn_med=float(np.median(bayar_all)),
                            premi_musim_med=float(np.median(prem_all)),
                            he_med=float(np.median(he_all)),
                            he_q1=float(np.percentile(he_all, 25)),
                            he_q3=float(np.percentile(he_all, 75)),
                            pct_he_negatif=float(100 * np.mean(np.array(he_all) < 0)),
                            fn_med=float(np.median(fn_all)),
                            ce_gain_med=float(np.median(ce_all)),
                            pct_ce_negatif=float(100 * np.mean(np.array(ce_all) < 0))))
    tab = pd.DataFrame(ringkas)
    tabz = pd.DataFrame(detail_zona)
    tab.to_csv(f"{OUT}/tab_adu_rancangan.csv", index=False)
    tabz.to_csv(f"{OUT}/tab_adu_rancangan_per_zona.csv", index=False)

    bid = pd.Series(waktu).dt.to_period("M").astype(str).values
    stat_zona = {}
    for z in [1, 3, 5]:
        kol = np.flatnonzero(zona == z)
        seri = np.nanmean(mat_pes[:, kol], axis=1)
        u = float(np.nanpercentile(seri, 90))
        mulai, dur = episode(seri, u, jeda=2)
        hari_b = pd.Series((seri > u).astype(float)).groupby(bid).sum().values
        ep_b = pd.Series(0.0, index=pd.unique(bid))
        if len(mulai):
            vc = pd.Series(bid[mulai]).value_counts()
            ep_b.loc[vc.index] = vc.values.astype(float)
        ep_b = ep_b.values
        q = 1 / dur.mean()
        ks = stats.kstest(dur, lambda x: stats.geom.cdf(x, q))
        t_mid = pd.DataFrame({"w": waktu, "b": bid}).groupby("b")["w"].apply(
            lambda s: (s.mean() - waktu.min()).days).reindex(pd.unique(bid)).values.astype(float)
        pjg = pd.Series(1.0, index=range(len(bid))).groupby(bid).sum().reindex(pd.unique(bid)).values
        glm = {}
        for J in (0, 1, 2):
            cols = [np.ones_like(t_mid)]
            for j in range(1, J + 1):
                cols += [np.cos(2*np.pi*j*t_mid/T), np.sin(2*np.pi*j*t_mid/T)]
            X = np.column_stack(cols)
            m = sm.GLM(ep_b, X, family=sm.families.Poisson(), offset=np.log(pjg)).fit()
            glm[f"J={J}"] = round(float(m.aic), 2)
        lr = None
        cols0 = np.ones((len(t_mid), 1))
        m0 = sm.GLM(ep_b, cols0, family=sm.families.Poisson(), offset=np.log(pjg)).fit()
        cols2 = np.column_stack([np.ones_like(t_mid)] +
                                [f(2*np.pi*j*t_mid/T) for j in (1, 2) for f in (np.cos, np.sin)])
        m2 = sm.GLM(ep_b, cols2, family=sm.families.Poisson(), offset=np.log(pjg)).fit()
        lr = 2 * (m2.llf - m0.llf)
        stat_zona[f"zona {z}"] = dict(
            u_p90=round(u, 2), n_episode=int(len(dur)),
            episode_per_thn=round(len(dur)/n_thn, 2),
            durasi_rata=round(float(dur.mean()), 2),
            durasi_maks=int(dur.max()),
            extremal_index=round(float(q), 3),
            rasio_var_mean_HARI=round(float(hari_b.var(ddof=1)/hari_b.mean()), 3),
            rasio_var_mean_EPISODE=round(float(ep_b.var(ddof=1)/ep_b.mean()), 3),
            geom_ks_p=round(float(ks.pvalue), 4),
            aic=glm, LR_stat=round(float(lr), 2),
            LR_p=round(float(1 - stats.chi2.cdf(lr, 4)), 5))
    hasil["diagnostik_ambang_relatif"] = stat_zona

    with open(f"{OUT}/hasil_basisrisk5.json", "w") as f:
        json.dump(dict(ringkas=ringkas, per_zona=detail_zona, **hasil), f,
                  indent=2, default=float)

    pd.set_option("display.width", 240)
    print("=== ADU RANCANGAN (agregat lintas zona pesisir) ===")
    print(tab.round(3).to_string(index=False))
    print("\n=== PER ZONA ===")
    print(tabz.round(2).to_string(index=False))
    print("\n=== DIAGNOSTIK pada ambang relatif p90 ===")
    print(json.dumps(stat_zona, indent=2, default=float))


if __name__ == "__main__":
    main()
