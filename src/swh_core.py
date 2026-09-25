"""shared ERA5 helpers"""
from __future__ import annotations

import datetime as _dt
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DATASET = "derived-era5-single-levels-daily-statistics"
VARIABLE = "significant_height_of_combined_wind_waves_and_swell"
TIME_ZONE = "utc+07:00"

WPP_BBOX = {
    "573": [-7.0, 104.0, -15.0, 128.0],
    "572": [6.0, 93.0, -8.0, 106.0],
    "712": [-3.0, 105.0, -7.5, 117.0],
    "713": [-1.0, 116.0, -9.0, 123.0],
    "718": [-5.0, 130.0, -11.0, 141.0],
}

WPP_NAMA = {
    "573": "Samudra Hindia selatan Jawa–NTT & Laut Sawu",
    "572": "Samudra Hindia barat Sumatera & Selat Sunda",
    "712": "Laut Jawa",
    "713": "Selat Makassar, Laut Flores, Teluk Bone",
    "718": "Laut Aru & Laut Arafura",
}

AMBANG_TIER = {
    "<5 GT": 2.0,
    "5-30 GT": 2.5,
    ">30 GT": 4.0,
}

MUSIM = {
    12: "Barat", 1: "Barat", 2: "Barat",
    3: "Peralihan I", 4: "Peralihan I", 5: "Peralihan I",
    6: "Timur", 7: "Timur", 8: "Timur",
    9: "Peralihan II", 10: "Peralihan II", 11: "Peralihan II",
}

INDEKS_RESMI = "swh_mean"


N_TAHUN_DEFAULT = 5
TAHUN_MULAI_PENUH = 1995


def tahun_lengkap_terakhir(n=N_TAHUN_DEFAULT, hari_ini=None):
    hari_ini = hari_ini or _dt.date.today()
    akhir = hari_ini.year - 1
    return akhir - n + 1, akhir


def tahun_dari_nama(path):
    m = re.findall(r"(19|20)\d{2}", Path(path).stem)
    if not m:
        return None
    return int(re.findall(r"((?:19|20)\d{2})", Path(path).stem)[0])


def open_any(path):
    path = Path(path)
    if zipfile.is_zipfile(path):
        extract_dir = path.with_suffix("")
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(extract_dir)
        ncs = sorted(extract_dir.glob("*.nc"))
        if not ncs:
            raise RuntimeError(f"tidak ada .nc di dalam {path}")
        if len(ncs) > 1:
            return xr.open_mfdataset(ncs, combine="by_coords")
        return xr.open_dataset(ncs[0])
    return xr.open_dataset(path)


def pick_var(ds):
    for cand in ("swh", VARIABLE):
        if cand in ds.data_vars:
            return ds[cand]
    numeric = [v for v in ds.data_vars if ds[v].ndim >= 2]
    if len(numeric) == 1:
        return ds[numeric[0]]
    raise KeyError(f"tidak menemukan variabel SWH. data_vars = {list(ds.data_vars)}")


def pick_dims(da):
    tname = next((d for d in ("valid_time", "time", "date") if d in da.dims), None)
    yname = next((d for d in ("latitude", "lat") if d in da.dims), None)
    xname = next((d for d in ("longitude", "lon") if d in da.dims), None)
    if not all([tname, yname, xname]):
        raise KeyError(f"dimensi tak dikenali: {da.dims}")
    return tname, yname, xname


def reduce_spatial(da):
    tname, yname, xname = pick_dims(da)
    da = da.squeeze(drop=True)

    w = np.cos(np.deg2rad(da[yname]))
    w = w.where(da.notnull()).fillna(0)

    mean = (da * w).sum(dim=(yname, xname)) / w.sum(dim=(yname, xname))
    mx = da.max(dim=(yname, xname))
    p90 = da.quantile(0.90, dim=(yname, xname))
    ngrid = da.notnull().sum(dim=(yname, xname))

    df = pd.DataFrame({
        "tanggal": pd.to_datetime(da[tname].values).normalize(),
        "swh_mean": np.asarray(mean, dtype="float64"),
        "swh_p90": np.asarray(p90, dtype="float64"),
        "swh_max": np.asarray(mx, dtype="float64"),
        "n_grid": np.asarray(ngrid, dtype="int64"),
    })
    return df.dropna(subset=["swh_mean"]).sort_values("tanggal").reset_index(drop=True)


def tambah_kalender(df):
    df = df.copy()
    df["tahun"] = df["tanggal"].dt.year
    df["bulan"] = df["tanggal"].dt.month
    df["musim"] = df["bulan"].map(MUSIM)
    return df


def build_series(files, indeks=INDEKS_RESMI):
    frames = []
    for f in sorted(files):
        with open_any(f) as ds:
            frames.append(reduce_spatial(pick_var(ds)))
    if not frames:
        raise RuntimeError("tidak ada file untuk diproses")
    df = (pd.concat(frames)
            .drop_duplicates("tanggal")
            .sort_values("tanggal")
            .reset_index(drop=True))
    return tambah_kalender(df)


def cacah_bulanan(df, ambang, kolom=INDEKS_RESMI):
    d = df.copy()
    d["terpicu"] = (d[kolom] > ambang).astype(int)
    out = (d.groupby(["tahun", "bulan"])
             .agg(n_hari=("terpicu", "size"),
                  n_terpicu=("terpicu", "sum"),
                  swh_rata=(kolom, "mean"),
                  swh_maks=(kolom, "max"))
             .reset_index())
    out["musim"] = out["bulan"].map(MUSIM)
    out["proporsi_terpicu"] = out["n_terpicu"] / out["n_hari"]
    return out


def ringkas_ambang(df, ambang_list=(1.5, 2.0, 2.5, 3.0, 4.0), kolom=INDEKS_RESMI):
    n_tahun = df["tahun"].nunique()
    rows = []
    for a in ambang_list:
        n = int((df[kolom] > a).sum())
        rows.append({
            "ambang_m": a,
            "n_hari_terpicu": n,
            "persen_hari": 100 * n / len(df),
            "hari_per_tahun": n / n_tahun,
        })
    return pd.DataFrame(rows)
