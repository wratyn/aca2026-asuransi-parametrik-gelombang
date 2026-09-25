#!/usr/bin/env python3
"""download ERA5 daily max SWH from CDS"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swh_core as core

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]


def build_request(year, area, months=None, days=None):
    return {
        "product_type": "reanalysis",
        "variable": [core.VARIABLE],
        "year": str(year),
        "month": months or MONTHS,
        "day": days or DAYS,
        "daily_statistic": "daily_maximum",
        "time_zone": core.TIME_ZONE,
        "frequency": "1_hourly",
        "area": area,
    }


def download_year(client, wpp, year, area, outdir):
    target = outdir / f"era5_swh_wpp{wpp}_{year}.nc"
    if target.exists() and target.stat().st_size > 0:
        print(f"  [skip] {target.name} sudah ada")
        return target
    print(f"  [get ] {target.name} ... (antre di CDS, bisa 1-30 menit)")
    client.retrieve(core.DATASET, build_request(year, area), str(target))
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wpp", default="573", choices=sorted(core.WPP_BBOX))
    ap.add_argument("--tahun", type=int, default=core.N_TAHUN_DEFAULT,
                    help=f"banyak tahun terakhir (default {core.N_TAHUN_DEFAULT})")
    ap.add_argument("--penuh", action="store_true",
                    help=f"abaikan --tahun, unduh sejak {core.TAHUN_MULAI_PENUH}")
    ap.add_argument("--start", type=int, default=None, help="timpa tahun awal")
    ap.add_argument("--end", type=int, default=None, help="timpa tahun akhir")
    ap.add_argument("--test", action="store_true",
                    help="unduh 1 bulan saja (Juli 2020) untuk uji koneksi")
    args = ap.parse_args()

    awal, akhir = core.tahun_lengkap_terakhir(args.tahun)
    if args.penuh:
        awal = core.TAHUN_MULAI_PENUH
    if args.start is not None:
        awal = args.start
    if args.end is not None:
        akhir = args.end
    if awal > akhir:
        sys.exit(f"Rentang tahun tidak masuk akal: {awal}-{akhir}")

    try:
        import cdsapi
    except ImportError:
        sys.exit('cdsapi belum terpasang. Jalankan: pip install "cdsapi>=0.7.7" xarray netCDF4')

    if not (Path.home() / ".cdsapirc").exists():
        sys.exit("~/.cdsapirc tidak ditemukan. Lihat https://cds.climate.copernicus.eu/how-to-api")

    RAW.mkdir(parents=True, exist_ok=True)
    area = core.WPP_BBOX[args.wpp]
    client = cdsapi.Client()

    if args.test:
        target = RAW / f"TEST_era5_swh_wpp{args.wpp}_202007.nc"
        print(f"Uji koneksi: WPP {args.wpp}, Juli 2020, area={area}")
        if not target.exists():
            client.retrieve(core.DATASET, build_request(2020, area, months=["07"]), str(target))
        with core.open_any(target) as ds:
            da = core.pick_var(ds)
            print("\nDimensi :", dict(da.sizes))
            print(core.reduce_spatial(da).describe())
        print(f"\nKoneksi CDS OK. Lanjut: python src/download_era5_swh.py --wpp {args.wpp}")
        return

    years = list(range(awal, akhir + 1))
    label = "penuh" if args.penuh else f"{len(years)} tahun terakhir"
    print(f"WPP {args.wpp} — {core.WPP_NAMA.get(args.wpp, '')}")
    print(f"Jendela : {awal}-{akhir} ({label}, {len(years)} request)")
    print(f"Area    : {area}\n")

    for y in years:
        download_year(client, args.wpp, y, area, RAW)

    print(f"\nSelesai. Lanjut olah jadi CSV:")
    print(f"  python src/proses_swh.py --wpp {args.wpp} --start {awal} --end {akhir}")


if __name__ == "__main__":
    main()
