#!/usr/bin/env python3
"""check ERA5 files are complete"""
import argparse
import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wpp", default="573")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    args = ap.parse_args()

    baris, hilang, bentuk = [], [], set()

    for th in range(args.start, args.end + 1):
        f = RAW / f"era5_swh_wpp{args.wpp}_{th}.nc"
        if not f.exists() or f.stat().st_size == 0:
            hilang.append(th)
            continue
        try:
            da = xr.open_dataset(f)["swh"]
        except Exception as e:
            baris.append({"tahun": th, "status": f"GAGAL DIBUKA: {type(e).__name__}"})
            continue

        t = pd.to_datetime(da["valid_time"].values)
        harap = 366 if calendar.isleap(th) else 365
        nilai = da.values
        n_darat = int(np.isnan(nilai).all(axis=0).sum())

        masalah = []
        if len(t) != harap:
            masalah.append(f"{len(t)} hari (harusnya {harap})")
        if t.duplicated().any():
            masalah.append("ada tanggal ganda")
        if (pd.Series(t).diff().dt.days.dropna() != 1).any():
            masalah.append("ada tanggal bolong")
        if t.min().year != th or t.max().year != th:
            masalah.append("ada tanggal di luar tahunnya")

        bentuk.add((da.sizes["latitude"], da.sizes["longitude"]))
        baris.append({
            "tahun": th,
            "hari": len(t),
            "grid": f"{da.sizes['latitude']}x{da.sizes['longitude']}",
            "sel_darat": n_darat,
            "mb": round(f.stat().st_size / 1e6, 2),
            "status": "ok" if not masalah else "; ".join(masalah),
        })
        da.close()

    if baris:
        print(pd.DataFrame(baris).to_string(index=False))

    print()
    if hilang:
        print(f"BELUM ADA ({len(hilang)} tahun): {hilang}")
        print("  Unduh dengan:")
        print(f"  python src/download_era5_swh.py --wpp {args.wpp} "
              f"--start {min(hilang)} --end {max(hilang)}")
    else:
        print(f"Lengkap: {args.start}-{args.end} ({args.end - args.start + 1} tahun).")

    if len(bentuk) > 1:
        print(f"\nPERINGATAN: bentuk grid tidak seragam -> {bentuk}")
        print("  Kemungkinan bounding box berbeda antar akun. Samakan dulu sebelum digabung.")

    buruk = [b for b in baris if b.get("status") != "ok"]
    sys.exit(1 if (hilang or buruk or len(bentuk) > 1) else 0)


if __name__ == "__main__":
    main()
