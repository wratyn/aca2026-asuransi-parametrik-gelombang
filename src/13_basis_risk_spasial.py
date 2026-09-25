#!/usr/bin/env python3
"""spatial basis risk: zone index vs each coastal cell"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR/"src"))
import zonasi as Z

KUNCI  = {1:"573-A", 4:"573-D", 6:"573-F"}
PERSEN = {1:85, 2:90, 3:95}

def main():
    h = Z.jalankan(simpan=False, verbose=False)
    mat, label, lat, lon = h["mat"], h["label"], h["lat"], h["lon"]
    baris, per_sel = [], []
    for zn, kunci in KUNCI.items():
        sel = label == zn
        M = mat[:, sel]
        idx = M.mean(axis=1)
        for tier, p in PERSEN.items():
            u_z = np.percentile(idx, p)
            Xz = (idx > u_z).astype(int)
            fn = fp = he = 0.0
            fns, fps, hes = [], [], []
            for c in range(M.shape[1]):
                u_i = np.percentile(M[:, c], p)
                Xi = (M[:, c] > u_i).astype(int)
                fnc = np.mean((Xi == 1) & (Xz == 0))
                fpc = np.mean((Xz == 1) & (Xi == 0))
                hec = 1 - np.var(Xi - Xz)/np.var(Xi)
                fns.append(fnc); fps.append(fpc); hes.append(hec)
                per_sel.append(dict(zona=kunci, tier=tier, lat=lat[sel][c], lon=lon[sel][c],
                                    u_sel=round(float(u_i),4), false_negative=round(fnc,4),
                                    false_positive=round(fpc,4), HE=round(hec,4)))
            baris.append(dict(zona=kunci, tier=tier, persentil=p, u_zona=round(float(u_z),4),
                              n_sel=int(sel.sum()),
                              false_negative=round(float(np.mean(fns)),4),
                              false_positive=round(float(np.mean(fps)),4),
                              HE_rata=round(float(np.mean(hes)),4),
                              HE_p10=round(float(np.percentile(hes,10)),4),
                              HE_min=round(float(np.min(hes)),4)))
    rs = pd.DataFrame(baris); rs.to_csv(AKAR/"output/tables/basis_risk_ringkas.csv", index=False)
    pd.DataFrame(per_sel).to_csv(AKAR/"data/processed/basis_risk_per_sel.csv", index=False)

    print("BASIS RISK SPASIAL — indeks zona vs kondisi di sel pesisir masing-masing")
    print("25 tahun, 9.131 hari, 75 sel pesisir pada tiga zona cakupan\n")
    print(f"{'zona':<8} {'tier':<5} {'u_zona':>8} {'FN':>7} {'FP':>7} {'HE rata':>9} {'HE p10':>8} {'HE min':>8}")
    print("-"*64)
    for _, r in rs.iterrows():
        print(f"{r.zona:<8} {r.tier:<5} {r.u_zona:>8.3f} {r.false_negative:>7.3f} "
              f"{r.false_positive:>7.3f} {r.HE_rata:>9.3f} {r.HE_p10:>8.3f} {r.HE_min:>8.3f}")
    print("\n-> output/tables/basis_risk_ringkas.csv, data/processed/basis_risk_per_sel.csv")

if __name__ == "__main__":
    main()
