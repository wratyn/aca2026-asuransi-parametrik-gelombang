#!/usr/bin/env python3
"""diff tables with tolerance. usage: python tools/cek_reproduksi.py REF_DIR [NEW_DIR]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

RTOL, ATOL = 1e-6, 1e-9


def _json(a, b):
    ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
    def rata(x, p=""):
        if isinstance(x, dict):
            return {k2: v2 for k, v in x.items() for k2, v2 in rata(v, f"{p}{k}.").items()}
        if isinstance(x, list):
            return {k2: v2 for i, v in enumerate(x) for k2, v2 in rata(v, f"{p}{i}.").items()}
        return {p[:-1]: x}
    fa, fb = rata(ja), rata(jb)
    if fa.keys() != fb.keys():
        return "BEDA", "kunci JSON berbeda"
    beda = [k for k in fa if not (fa[k] == fb[k] or (isinstance(fa[k], (int, float)) and isinstance(fb[k], (int, float))
                                                    and np.isclose(fa[k], fb[k], rtol=RTOL, atol=ATOL)))]
    return ("SETARA", "") if not beda else ("BEDA", f"{len(beda)} nilai: {beda[:3]}")


def banding_berkas(a, b):
    a, b = Path(a), Path(b)
    if a.read_bytes() == b.read_bytes():
        return "IDENTIK", ""
    if a.suffix == ".json":
        return _json(a, b)
    A, B = pd.read_csv(a, low_memory=False), pd.read_csv(b, low_memory=False)
    if A.shape != B.shape:
        return "BEDA", f"bentuk {A.shape} vs {B.shape}"
    if list(A.columns) != list(B.columns):
        return "BEDA", f"kolom berbeda: {sorted(set(A.columns) ^ set(B.columns))}"
    kat, catatan, n_sel = "SETARA", [], A.size
    for c in A.columns:
        x, y = A[c], B[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            ok = np.isclose(xv, yv, rtol=RTOL, atol=ATOL, equal_nan=True)
            if not ok.all():
                d = np.abs(xv - yv)[~ok]
                if d.max() <= 1.0001e-4 and (~ok).sum() <= max(1, 0.01 * len(xv)):
                    kat = "PEMBULATAN" if kat == "SETARA" else kat
                    catatan.append(f"{c}: {(~ok).sum()} sel, |d|<={d.max():.1g}")
                else:
                    kat = "BEDA"
                    catatan.append(f"{c}: {(~ok).sum()} sel, maks|d|={d.max():.3g}")
        else:
            ne = ~((x.astype(str) == y.astype(str)) | (x.isna() & y.isna()))
            if ne.any():
                kat = "BEDA"; catatan.append(f"{c}: {ne.sum()} teks berbeda")
    return kat, "; ".join(catatan)


def bandingkan_folder(ref, baru, subfolder):
    ref, baru = Path(ref), Path(baru)
    hitung, masalah = {}, []
    print(f"\n{'KATEGORI':<11} BERKAS")
    for sf in subfolder:
        f_ref = {p.relative_to(ref) for p in (ref / sf).rglob("*") if p.suffix in (".csv", ".json")}
        f_baru = {p.relative_to(baru) for p in (baru / sf).rglob("*") if p.suffix in (".csv", ".json")}
        for f in sorted(f_ref | f_baru):
            if f not in f_baru:
                kat, cat = "HILANG", "tidak dihasilkan ulang"
            elif f not in f_ref:
                kat, cat = "BARU", "tidak ada di rujukan"
            else:
                kat, cat = banding_berkas(baru / f, ref / f)
            hitung[kat] = hitung.get(kat, 0) + 1
            if kat not in ("IDENTIK", "SETARA"):
                masalah.append(kat)
            print(f"{kat:<11} {f}" + (f"   [{cat}]" if cat else ""))
    print("\nringkasan: " + ", ".join(f"{k} {v}" for k, v in sorted(hitung.items())))
    ok = not any(k in ("BEDA", "HILANG") for k in masalah)
    print("REPRODUKSI BERHASIL" if ok else "ADA PERBEDAAN — periksa baris BEDA/HILANG di atas")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ref = sys.argv[1]
    baru = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent
    sys.exit(0 if bandingkan_folder(ref, baru, ["data/dim", "data/processed", "output/tables"]) else 2)
