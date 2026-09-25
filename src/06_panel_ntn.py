#!/usr/bin/env python3
"""stack BPS NTN files into one monthly panel"""
import glob
from pathlib import Path
import pandas as pd

AKAR = Path(__file__).resolve().parent.parent
fs = sorted(glob.glob(str(AKAR / "data/raw/bps_ntn/ntn_*.csv")))
p = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
p = p.sort_values(["provinsi", "tahun", "bulan"], kind="stable").reset_index(drop=True)
assert not p.duplicated(["provinsi", "tahun", "bulan"]).any(), "bulan ganda"
p.to_csv(AKAR / "data/processed/panel_ntn.csv", index=False)
print(p.groupby("provinsi").agg(n_bulan=("ntn", "size"), mulai=("tahun", "min"),
                                 akhir=("tahun", "max")).to_string())
print(f"-> panel_ntn.csv ({len(p)} baris)")
