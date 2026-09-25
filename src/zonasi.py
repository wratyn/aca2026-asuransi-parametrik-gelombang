"""correlation-based zoning of coastal cells"""
from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

AKAR = Path(__file__).resolve().parent.parent

RAW = AKAR / "data" / "raw"
PROCESSED = AKAR / "data" / "processed"
DIM = AKAR / "data" / "dim"
TAB = AKAR / "output" / "tables"
FIG = AKAR / "output" / "figures"

POLA_ERA5 = "era5_swh_wpp573_20*.nc"

PALET = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9",
         "#7A5195", "#3C6E47"]

MUSIM = {12: "Barat", 1: "Barat", 2: "Barat",
         3: "Peralihan I", 4: "Peralihan I", 5: "Peralihan I",
         6: "Timur", 7: "Timur", 8: "Timur",
         9: "Peralihan II", 10: "Peralihan II", 11: "Peralihan II"}

K_MIN, K_MAKS = 2, 12

AMBANG_KETERWAKILAN = 0.75  # min p10 corr of a cell to its zone index
JARI_BUMI_KM = 6371.0


def muat(pola=None):
    pola = str(pola) if pola is not None else str(RAW / POLA_ERA5)
    fs = sorted(glob.glob(pola))
    if not fs:
        raise FileNotFoundError(
            f"tidak ada berkas ERA5 yang cocok dengan pola:\n  {pola}\n"
            f"akar proyek terdeteksi di: {AKAR}\n"
            f"folder data/raw ada? {RAW.is_dir()}  "
            f"(isi: {sorted(x.name for x in RAW.glob('*.nc'))[:5] if RAW.is_dir() else '-'})")
    ds = xr.concat([xr.open_dataset(f) for f in fs], dim="valid_time")
    return ds["swh"].load().rename({"valid_time": "waktu"})


def klasifikasi(da, sertakan_tepi=False):
    laut = da.notnull().all("waktu").values
    darat = ~laut
    nlat, nlon = laut.shape
    pes = np.zeros_like(laut)
    for i in range(nlat):
        for j in range(nlon):
            if not laut[i, j]:
                continue
            dekat_darat = darat[max(0, i - 1):i + 2, max(0, j - 1):j + 2].any()
            di_tepi = i in (0, nlat - 1) or j in (0, nlon - 1)
            if dekat_darat or (sertakan_tepi and di_tepi):
                pes[i, j] = True
    return laut, pes


def terpapar_samudra(da, laut):
    darat = ~laut
    lat = da.latitude.values
    nlat, nlon = laut.shape
    urut_selatan = np.argsort(lat)
    out = np.zeros_like(laut)
    for j in range(nlon):
        ada_darat = False
        for i in urut_selatan:
            if darat[i, j]:
                ada_darat = True
            elif laut[i, j] and not ada_darat:
                out[i, j] = True
    return out


def matriks(da, mask):
    v = da.values
    idx = np.argwhere(mask)
    return np.stack([v[:, i, j] for i, j in idx], axis=1), idx


def jarak_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 2 * JARI_BUMI_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def anomali(mat, waktu):
    doy = pd.Series(waktu).dt.dayofyear.values
    klim = np.zeros_like(mat)
    for d in range(1, 367):
        m = doy == d
        if m.any():
            klim[m] = np.nanmean(mat[m], axis=0)
    tabel = np.stack([np.nanmean(mat[doy == d], axis=0) if (doy == d).any()
                      else np.full(mat.shape[1], np.nan) for d in range(1, 367)])
    tabel = pd.DataFrame(tabel).interpolate(limit_direction="both").values
    pad = np.vstack([tabel[-15:], tabel, tabel[:15]])
    halus = pd.DataFrame(pad).rolling(31, center=True, min_periods=1).mean().values[15:-15]
    for d in range(1, 367):
        m = doy == d
        if m.any():
            klim[m] = halus[d - 1]
    return mat - klim


def korelasi(mat):
    x = mat - mat.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return np.clip((x / s).T @ (x / s) / len(x), -1.0, 1.0)


def peluruhan(D, R, n_bin=30, d_maks=1500.0):
    iu = np.triu_indices_from(D, k=1)
    d, r = D[iu], R[iu]
    pakai = d <= d_maks
    d, r = d[pakai], r[pakai]
    tepi = np.linspace(0, d.max(), n_bin + 1)
    pusat, rata = [], []
    for a, b in zip(tepi[:-1], tepi[1:]):
        m = (d >= a) & (d < b)
        if m.sum() > 20:
            pusat.append((a + b) / 2)
            rata.append(r[m].mean())
    pusat, rata = np.array(pusat), np.array(rata)

    def f(x, d0, nug):
        return (1 - nug) * np.exp(-x / d0) + nug

    try:
        p, _ = curve_fit(f, pusat, rata, p0=[400.0, 0.2],
                         bounds=([50.0, 0.0], [3000.0, 0.9]))
    except Exception:
        p = [np.nan, np.nan]
    return float(p[0]), float(p[1]), pusat, rata


RADIUS_TETANGGA_KM = 120.0


def konektivitas(D, radius=RADIUS_TETANGGA_KM):
    A = ((D > 0) & (D <= radius)).astype(float)
    n_komponen, _ = connected_components(coo_matrix(A), directed=False)
    return coo_matrix(A), int(n_komponen)


def baku(mat):
    x = mat - mat.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return (x / s).T


def _cluster(X, conn, k):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*connected components.*")
        return AgglomerativeClustering(n_clusters=k, linkage="ward",
                                       connectivity=conn).fit_predict(X)


def keterwakilan_per_sel(mat, label):
    out = np.full(mat.shape[1], np.nan)
    for z in set(label):
        m = label == z
        ix = np.nanmean(mat[:, m], axis=1)
        for c in np.flatnonzero(m):
            out[c] = np.corrcoef(mat[:, c], ix)[0, 1]
    return out


def keterwakilan(mat, label):
    out = []
    for z in set(label):
        m = label == z
        ix = np.nanmean(mat[:, m], axis=1)
        for c in np.flatnonzero(m):
            out.append(np.corrcoef(mat[:, c], ix)[0, 1])
    return np.array(out)


def urutkan_barat_ke_timur(label, lon):
    urut = pd.Series(lon).groupby(label).mean().sort_values().index
    peta = {lama: baru + 1 for baru, lama in enumerate(urut)}
    return np.array([peta[x] for x in label])


def pilih_k(mat, X, R, conn, lon, ks=range(K_MIN, K_MAKS + 1)):
    Dc = 1.0 - R
    np.fill_diagonal(Dc, 0.0)
    baris, label_per_k = [], {}
    for k in ks:
        lab = urutkan_barat_ke_timur(_cluster(X, conn, k), lon)
        label_per_k[k] = lab
        sil = silhouette_score(Dc, lab, metric="precomputed")
        sama = lab[:, None] == lab[None, :]
        iu = np.triu_indices_from(R, k=1)
        intra = R[iu][sama[iu]].mean()
        inter = R[iu][~sama[iu]].mean()
        ukuran = pd.Series(lab).value_counts()
        rep = keterwakilan(mat, lab)
        baris.append(dict(k=k, silhouette=sil, korelasi_intra=intra,
                          korelasi_inter=inter, selisih=intra - inter,
                          keterwakilan_rata=rep.mean(),
                          keterwakilan_p10=float(np.percentile(rep, 10)),
                          keterwakilan_min=rep.min(),
                          sel_min=int(ukuran.min()), sel_maks=int(ukuran.max())))
    return pd.DataFrame(baris), label_per_k


def ringkas_zona(mat, idx, lat, lon, label, waktu, R, terpapar=None):
    bulan = pd.Series(waktu).dt.month.values
    baris = []
    for z in sorted(set(label)):
        kol = np.flatnonzero(label == z)
        seri = np.nanmean(mat[:, kol], axis=1)
        sub = R[np.ix_(kol, kol)]
        iu = np.triu_indices_from(sub, k=1)
        siklus = {MUSIM[b]: float(np.nanmean(seri[bulan == b])) for b in range(1, 13)}
        per_musim = pd.Series({m: np.nanmean([v for b, v in
                                              zip(range(1, 13), [np.nanmean(seri[bulan == bb])
                                                                 for bb in range(1, 13)])
                                              if MUSIM[b] == m]) for m in
                               ("Barat", "Peralihan I", "Timur", "Peralihan II")})
        u = float(np.nanpercentile(seri, 90))
        baris.append(dict(
            zona=int(z), n_sel=len(kol),
            lat_min=float(lat[kol].min()), lat_maks=float(lat[kol].max()),
            lon_min=float(lon[kol].min()), lon_maks=float(lon[kol].max()),
            lon_pusat=float(lon[kol].mean()), lat_pusat=float(lat[kol].mean()),
            swh_rata=float(np.nanmean(seri)), swh_p90=u, ambang_m=u,
            swh_maks=float(np.nanmax(seri)),
            korelasi_intra=float(sub[iu].mean()) if len(iu[0]) else np.nan,
            swh_musim_barat=float(per_musim["Barat"]),
            swh_musim_timur=float(per_musim["Timur"]),
            rasio_barat_timur=float(per_musim["Barat"] / per_musim["Timur"]),
            hari_terpicu_per_tahun=float((seri > u).sum() / (len(seri) / 365.25)),
            frac_terpapar=(float(np.mean(terpapar[kol])) if terpapar is not None else np.nan),
        ))
    out = pd.DataFrame(baris)
    rel = out.swh_rata / out.swh_rata.max()
    out["watak"] = np.where(rel >= 0.75, "terpapar Samudra Hindia",
                    np.where(rel >= 0.45, "peralihan", "terlindung / laut dalam"))
    return out


def siklus_bulanan(mat, label, waktu):
    bulan = pd.Series(waktu).dt.month.values
    out = []
    for z in sorted(set(label)):
        seri = np.nanmean(mat[:, np.flatnonzero(label == z)], axis=1)
        rata = np.nanmean(seri)
        for b in range(1, 13):
            v = float(np.nanmean(seri[bulan == b]))
            out.append(dict(zona=int(z), bulan=b, swh=v, swh_ternormalisasi=v / rata))
    return pd.DataFrame(out)


def gambar(da, laut, pes, idx, lat, lon, label, diag, k_pilih,
           dek_mentah, dek_anom, sik, ringkas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warna = {z: PALET[(z - 1) % len(PALET)] for z in sorted(set(label))}
    lonv, latv = da.longitude.values, da.latitude.values
    ABU_DARAT, ABU_LAUT, TINTA = "#c9c6c0", "#eeece8", "#4a4844"

    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1], wspace=0.2)

    ax = fig.add_subplot(gs[0, 0])
    ii, jj = np.where(laut & ~pes)
    ax.scatter(lonv[jj], latv[ii], s=15, c=ABU_LAUT, marker="s", label="laut lepas")
    ii, jj = np.where(~laut)
    ax.scatter(lonv[jj], latv[ii], s=15, c=ABU_DARAT, marker="s", label="darat")
    for z in sorted(set(label)):
        kol = np.flatnonzero(label == z)
        r = ringkas[ringkas.zona == z].iloc[0]
        ax.scatter(lon[kol], lat[kol], s=32, c=warna[z], marker="s",
                   edgecolors="white", linewidths=.4, zorder=3,
                   label=f"zona {z} — {r.watak} ({int(r.n_sel)} sel)")
    y0, y1 = latv.min() - .4, latv.max() + .4
    for z in sorted(set(label)):
        r = ringkas[ringkas.zona == z].iloc[0]
        ax.annotate(str(z), (r.lon_pusat, min(r.lat_pusat + 0.85, y1 - .45)),
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color=warna[z], zorder=5)
    ax.set_xlim(lonv.min() - .6, lonv.max() + .6)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Bujur (°BT)")
    ax.set_ylabel("Lintang (°LS)")
    ax.set_title(f"(a) {k_pilih} zona indeks hasil pengelompokan berkendala ketetanggaan\n"
                 f"sel pesisir WPP 573, SWH harian 2021–2025", fontsize=10.5, loc="left")
    ax.legend(fontsize=7.6, frameon=False, loc="lower left", ncol=2,
              handletextpad=.4, columnspacing=1.0)
    ax.grid(alpha=.22, linewidth=.6)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[0, 1])
    ax.axhspan(AMBANG_KETERWAKILAN, 1.0, color="#009E73", alpha=.07)
    ax.axhline(AMBANG_KETERWAKILAN, color="#009E73", linewidth=1.4, linestyle="--")
    ax.annotate(f"ambang {AMBANG_KETERWAKILAN:.2f}", (diag.k.max(), AMBANG_KETERWAKILAN),
                textcoords="offset points", xytext=(-4, 6), ha="right",
                fontsize=8.5, color="#009E73")
    ax.plot(diag.k, diag.keterwakilan_rata, "-o", color="#56B4E9", linewidth=2,
            markersize=5.5, label="rata-rata semua sel")
    ax.plot(diag.k, diag.keterwakilan_p10, "-o", color="#0072B2", linewidth=2.2,
            markersize=6, label="persentil ke-10 (sel terburuk)")
    yk = diag.set_index("k").keterwakilan_p10[k_pilih]
    ax.scatter([k_pilih], [yk], s=180, facecolor="none", edgecolor="#D55E00",
               linewidth=2.2, zorder=5)
    ax.annotate(f"k = {k_pilih}", (k_pilih, yk), textcoords="offset points",
                xytext=(9, -14), color="#D55E00", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Jumlah zona (k)")
    ax.set_ylabel("Korelasi sel terhadap indeks zonanya")
    ax.set_title("(b) Pemilihan jumlah zona:\nk terkecil yang memenuhi ambang",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.2, frameon=False, loc="lower right")
    ax.grid(alpha=.22, linewidth=.6)
    ax.set_axisbelow(True)

    fig.savefig(FIG / "fig_zonasi_peta.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5))

    ax = axs[0]
    for (pusat, rata, d0, nug), warna_g, nama in (
            (dek_mentah, "#56B4E9", "deret mentah (termasuk siklus musiman)"),
            (dek_anom, "#D55E00", "anomali (siklus musiman dibuang)")):
        ax.scatter(pusat, rata, s=22, color=warna_g, zorder=3, alpha=.85)
        xs = np.linspace(0, max(pusat), 200)
        ax.plot(xs, (1 - nug) * np.exp(-xs / d0) + nug, color=warna_g, linewidth=2,
                label=f"{nama}\n$d_0$ = {d0:.0f} km")
    ax.axvline(438, color=TINTA, linestyle="--", linewidth=1.3)
    ax.annotate("lebar zona lama\n(4° ≈ 438 km)", (438, .33),
                textcoords="offset points", xytext=(9, 0), fontsize=8.4, color=TINTA)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Jarak antar sel (km)")
    ax.set_ylabel("Korelasi SWH harian")
    ax.set_title("(a) Peluruhan korelasi spasial", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(.02, .02))
    ax.grid(alpha=.22, linewidth=.6); ax.set_axisbelow(True)

    ax = axs[1]
    for z in sorted(set(label)):
        sz = sik[sik.zona == z]
        r = ringkas[ringkas.zona == z].iloc[0]
        gaya = "-o" if r.frac_terpapar >= .5 else "--o"
        ax.plot(sz.bulan, sz.swh_ternormalisasi, gaya, color=warna[z], linewidth=2,
                markersize=4.2, label=f"zona {z}")
    ax.axhline(1.0, color="#b5b1ab", linewidth=1)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Bulan")
    ax.set_ylabel("SWH ÷ rerata tahunan zona")
    ax.set_title("(b) Bentuk siklus musiman per zona\n"
                 "garis penuh = mayoritas sel terpapar samudra", fontsize=10.5, loc="left")
    ax.set_ylim(0.42, 2.05)
    ax.legend(fontsize=7.8, frameon=False, ncol=3, loc="lower center",
              handletextpad=.4, columnspacing=1.1)
    ax.grid(alpha=.22, linewidth=.6); ax.set_axisbelow(True)

    ax = axs[2]
    d = diag.set_index("k")
    ax.plot(d.index, d.korelasi_intra, "-o", color="#009E73", linewidth=2,
            markersize=5.5, label="korelasi rerata DALAM zona")
    ax.plot(d.index, d.korelasi_inter, "-o", color="#CC79A7", linewidth=2,
            markersize=5.5, label="korelasi rerata ANTAR zona")
    ax.plot(d.index, d.silhouette, "-o", color="#E69F00", linewidth=2,
            markersize=5.5, label="silhouette (tidak stabil pada\nmedan gradien — tidak dipakai)")
    ax.axvline(k_pilih, color=TINTA, linestyle="--", linewidth=1.2)
    ax.annotate(f"k = {k_pilih}", (k_pilih, .95), textcoords="offset points",
                xytext=(6, 0), fontsize=9, color=TINTA, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Jumlah zona (k)")
    ax.set_ylabel("Nilai")
    ax.set_title("(c) Keterpisahan zona", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.grid(alpha=.22, linewidth=.6); ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG / "fig_zonasi_diagnostik.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def pilih_k_terkecil(diag, ambang=AMBANG_KETERWAKILAN):
    layak = diag[diag.keterwakilan_p10 >= ambang]
    if len(layak):
        return int(layak.k.min())
    return int(diag.loc[diag.keterwakilan_p10.idxmax(), "k"])


def jalankan(k_paksa=None, simpan=True, verbose=True):
    for p in (PROCESSED, DIM, TAB, FIG):
        Path(p).mkdir(parents=True, exist_ok=True)

    da = muat()
    waktu = pd.to_datetime(da.waktu.values)
    laut, pes = klasifikasi(da, sertakan_tepi=False)
    _, pes_lama = klasifikasi(da, sertakan_tepi=True)

    mat_laut, idx_laut = matriks(da, laut)
    mat, idx = matriks(da, pes)
    lonv, latv = da.longitude.values, da.latitude.values
    lon = np.array([lonv[j] for _, j in idx])
    lat = np.array([latv[i] for i, _ in idx])

    lon_l = np.array([lonv[j] for _, j in idx_laut])
    lat_l = np.array([latv[i] for i, _ in idx_laut])
    D_laut = jarak_km(lat_l, lon_l)
    d0, nugget, pusat_d, rata_r = peluruhan(D_laut, korelasi(mat_laut))
    anom_laut = anomali(mat_laut, waktu)
    d0a, nuga, pusat_a, rata_a = peluruhan(D_laut, korelasi(anom_laut))
    dek_mentah = (pusat_d, rata_r, d0, nugget)
    dek_anom = (pusat_a, rata_a, d0a, nuga)

    R = korelasi(mat)
    X = baku(mat)
    D_pes = jarak_km(lat, lon)
    conn, n_komponen = konektivitas(D_pes)

    diag, label_per_k = pilih_k(mat, X, R, conn, lon)
    k_pilih = int(k_paksa or pilih_k_terkecil(diag))
    label = label_per_k[k_pilih]

    tp_grid = terpapar_samudra(da, laut)
    terpapar = np.array([tp_grid[i, j] for i, j in idx])
    ringkas = ringkas_zona(mat, idx, lat, lon, label, waktu, R, terpapar)
    sik = siklus_bulanan(mat, label, waktu)
    sel = pd.DataFrame(dict(lat=lat, lon=lon, zona=label, terpapar=terpapar,
                            swh_rata=np.nanmean(mat, axis=0),
                            korelasi_thd_indeks=keterwakilan_per_sel(mat, label)))

    if simpan:
        diag.to_csv(TAB / "diagnostik_zonasi_wpp573.csv", index=False, float_format="%.5f")
        ringkas.to_csv(DIM / "dim_zona_wpp573.csv", index=False, float_format="%.5f")
        sel.to_csv(PROCESSED / "sel_zona_wpp573.csv", index=False, float_format="%.4f")
        sik.to_csv(TAB / "siklus_musiman_zona_wpp573.csv", index=False, float_format="%.5f")
        gambar(da, laut, pes, idx, lat, lon, label, diag, k_pilih,
               dek_mentah, dek_anom, sik, ringkas)
        with open(TAB / "zonasi_meta.json", "w") as f:
            json.dump(dict(k_terpilih=k_pilih, ambang_keterwakilan=AMBANG_KETERWAKILAN,
                           d0_km_deret_mentah=d0, nugget_mentah=nugget,
                           d0_km_anomali=d0a, nugget_anomali=nuga,
                           n_sel_pesisir=int(pes.sum()),
                           n_sel_pesisir_definisi_lama=int(pes_lama.sum()),
                           n_sel_laut=int(laut.sum()),
                           n_komponen_konektivitas=n_komponen,
                           radius_tetangga_km=RADIUS_TETANGGA_KM,
                           periode=[str(waktu.min().date()), str(waktu.max().date())]),
                      f, indent=2)

    if verbose:
        print(f"sel laut {laut.sum()} | pesisir benar {pes.sum()} "
              f"| pesisir definisi lama {pes_lama.sum()} "
              f"(selisih {pes_lama.sum() - pes.sum()} sel tepi kotak)")
        print(f"panjang korelasi d0: deret mentah {d0:.0f} km | anomali {d0a:.0f} km "
              f"(lebar zona lama 4 deg ~438 km)")
        print(f"graf ketetanggaan radius {RADIUS_TETANGGA_KM:.0f} km "
              f"-> {n_komponen} komponen (gugus pulau terpisah)")
        print(f"\nDiagnostik pemilihan k:\n{diag.round(4).to_string(index=False)}")
        print(f"\nk terpilih = {k_pilih} "
              f"(k terkecil dengan keterwakilan p10 >= {AMBANG_KETERWAKILAN})\n")
        print(ringkas.round(3).to_string(index=False))

    return dict(diag=diag, ringkas=ringkas, sel=sel, siklus=sik, label=label,
                k=k_pilih, d0=d0, nugget=nugget, d0_anomali=d0a,
                dek_mentah=dek_mentah, dek_anom=dek_anom, R=R, lat=lat, lon=lon,
                mat=mat, waktu=waktu, da=da, laut=laut, pes=pes, idx=idx,
                pusat_d=pusat_d, rata_r=rata_r)


if __name__ == "__main__":
    jalankan()


def zona_semua_sel(mat_laut, mat_pesisir, label_pesisir, idx_laut, n_haluskan=2):
    zona = sorted(set(label_pesisir))
    indeks = np.stack([np.nanmean(mat_pesisir[:, label_pesisir == z], axis=1)
                       for z in zona], axis=1)

    a = mat_laut - mat_laut.mean(axis=0, keepdims=True)
    b = indeks - indeks.mean(axis=0, keepdims=True)
    sa = a.std(axis=0, keepdims=True); sa[sa == 0] = 1.0
    sb = b.std(axis=0, keepdims=True); sb[sb == 0] = 1.0
    R = (a / sa).T @ (b / sb) / len(a)

    lab = np.array([zona[i] for i in R.argmax(axis=1)])
    rmaks = R.max(axis=1)

    if n_haluskan:
        pos = {(int(i), int(j)): k for k, (i, j) in enumerate(idx_laut)}
        for _ in range(n_haluskan):
            baru = lab.copy()
            for (i, j), k in pos.items():
                tetangga = [lab[pos[(i + di, j + dj)]]
                            for di in (-1, 0, 1) for dj in (-1, 0, 1)
                            if (di or dj) and (i + di, j + dj) in pos]
                if tetangga:
                    nilai, cacah = np.unique(tetangga, return_counts=True)
                    modus = nilai[cacah.argmax()]
                    if cacah.max() >= 5 and modus != lab[k]:
                        baru[k] = modus
            if (baru == lab).all():
                break
            lab = baru
    return lab, rmaks
