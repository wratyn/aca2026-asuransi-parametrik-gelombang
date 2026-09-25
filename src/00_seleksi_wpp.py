#!/usr/bin/env python3
"""WPP composite score: 0.50 hazard + 0.35 exposure + 0.15 production (min-max 0-100)"""
import sys
from pathlib import Path
import pandas as pd

AKAR = Path(__file__).resolve().parent.parent
MASUK = AKAR / "data/raw/seleksi_wpp/indikator_wpp.csv"
BOBOT = {"hari_swh_gt2_per_tahun": 0.50, "nelayan_2021": 0.35, "produksi_ton_2015_2020": 0.15}
NAMA = {"hari_swh_gt2_per_tahun": "skor_hazard", "nelayan_2021": "skor_exposure",
        "produksi_ton_2015_2020": "skor_produksi"}


def minmax(x):
    return 100 * (x - x.min()) / (x.max() - x.min())


def main():
    if not MASUK.exists():
        print(f"[lewati] {MASUK.relative_to(AKAR)} belum ada — tabel indikator per WPP belum "
              "dimasukkan ke repositori. Lihat docs/pemilihan-wpp.pdf untuk hasilnya.")
        return 0
    d = pd.read_csv(MASUK)
    hilang = set(BOBOT) - set(d.columns)
    if hilang:
        sys.exit(f"kolom hilang di indikator_wpp.csv: {sorted(hilang)}")
    for kol, nama in NAMA.items():
        d[nama] = minmax(d[kol].astype(float))
    d["composite_score"] = sum(w * d[NAMA[k]] for k, w in BOBOT.items())
    d = d.sort_values("composite_score", ascending=False).reset_index(drop=True)
    d.insert(0, "peringkat", range(1, len(d) + 1))
    d.round(1).to_csv(AKAR / "output/tables/seleksi_wpp.csv", index=False)
    print(d[["peringkat", "wpp", *NAMA.values(), "composite_score"]].round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
