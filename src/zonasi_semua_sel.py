#!/usr/bin/env python3
"""assign every sea cell to a zone (for the map)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "src"))
import zonasi as Z

h = Z.jalankan(simpan=False, verbose=False)
da, laut = h["da"], h["laut"]
mat_laut, idx_laut = Z.matriks(da, laut)
lab, rmaks = Z.zona_semua_sel(mat_laut, h["mat"], h["label"], idx_laut)
lonv, latv = da.longitude.values, da.latitude.values
out = pd.DataFrame(dict(lat=[latv[i] for i, _ in idx_laut], lon=[lonv[j] for _, j in idx_laut],
                        zona=lab, korelasi_thd_indeks=rmaks))
out.to_csv(Z.PROCESSED / "zona_semua_sel_wpp573.csv", index=False, float_format="%.4f")
print(f"{len(out)} sel laut -> {out.zona.nunique()} zona; tersimpan zona_semua_sel_wpp573.csv")
