#!/usr/bin/env python3
"""zone index, episodes, monthly counts"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "src"))
import zonasi as Z

KUNCI  = {1: "573-A", 4: "573-D", 6: "573-F"}
PERSEN = {1: 85, 2: 90, 3: 95}
JEDA   = 2  # calm days that split two episodes
T      = 365.25
MUSIM  = {12:"Barat",1:"Barat",2:"Barat",3:"Peralihan I",4:"Peralihan I",5:"Peralihan I",
          6:"Timur",7:"Timur",8:"Timur",9:"Peralihan II",10:"Peralihan II",11:"Peralihan II"}


def episode(x, u, jeda=JEDA):
    atas = x > u
    if not atas.any():
        return np.array([]), np.array([])
    d = np.diff(atas.astype(int))
    mulai = list(np.where(d == 1)[0] + 1)
    akhir = list(np.where(d == -1)[0])
    if atas[0]:  mulai.insert(0, 0)
    if atas[-1]: akhir.append(len(atas) - 1)
    m, a = [mulai[0]], []
    for i in range(1, len(mulai)):
        if mulai[i] - akhir[i-1] - 1 < jeda:
            continue
        a.append(akhir[i-1]); m.append(mulai[i])
    a.append(akhir[-1])
    return np.array(m), np.array(a)


def main():
    h = Z.jalankan(simpan=False, verbose=False)
    mat, label, waktu = h["mat"], h["label"], pd.to_datetime(h["waktu"])
    lat, lon = h["lat"], h["lon"]

    baris = []
    indeks = {}
    for zn, kunci in KUNCI.items():
        m = mat[:, label == zn]
        rata = m.mean(axis=1); maks = m.max(axis=1); p90 = np.percentile(m, 90, axis=1)
        indeks[kunci] = rata
        baris.append(pd.DataFrame({"tanggal": waktu.date, "zona": kunci,
                                   "swh_rerata": rata.round(4), "swh_maks": maks.round(4),
                                   "swh_p90": p90.round(4), "n_sel": (label == zn).sum()}))
    harian = pd.concat(baris, ignore_index=True)
    harian.to_csv(AKAR/"data/processed/swh_harian_zona.csv", index=False)

    dz = []
    for zn, kunci in KUNCI.items():
        sel = label == zn; m = mat[:, sel]
        dz.append(dict(zona=kunci, zona_num=zn, n_sel=int(sel.sum()),
                       lon_min=round(float(lon[sel].min()),2), lon_maks=round(float(lon[sel].max()),2),
                       lat_min=round(float(lat[sel].min()),2), lat_maks=round(float(lat[sel].max()),2),
                       swh_rerata=round(float(m.mean()),4),
                       **{f"u_p{p}": round(float(np.percentile(indeks[kunci], p)),4) for p in (85,90,95)}))
    pd.DataFrame(dz).to_csv(AKAR/"data/dim/dim_zona.csv", index=False)

    eps, cac = [], []
    for zn, kunci in KUNCI.items():
        x = indeks[kunci]
        for tier, p in PERSEN.items():
            u = float(np.percentile(x, p))
            m_, a_ = episode(x, u)
            for j, (i0, i1) in enumerate(zip(m_, a_), 1):
                t0, t1 = waktu[i0], waktu[i1]
                jeda_seb = int((i0 - a_[j-2] - 1)) if j > 1 else -1
                eps.append(dict(episode_id=f"{kunci}-{tier}-{j:04d}", zona=kunci, tier=tier,
                                tgl_mulai=t0.date(), tgl_akhir=t1.date(), durasi=int(i1-i0+1),
                                swh_puncak=round(float(x[i0:i1+1].max()),4),
                                swh_rerata_episode=round(float(x[i0:i1+1].mean()),4),
                                hari_dalam_tahun=round(float(t0.dayofyear),1),
                                musim=MUSIM[t0.month], tahun=int(t0.year),
                                jeda_sebelumnya=jeda_seb, u_k=round(u,4)))
            df = pd.DataFrame({"t": waktu, "x": x})
            df["thn"], df["bln"] = df.t.dt.year, df.t.dt.month
            hari = df.groupby(["thn","bln"]).size().rename("hari")
            terpicu = df.assign(p=(df.x>u).astype(int)).groupby(["thn","bln"]).p.sum().rename("n_hari_terpicu")
            aw = pd.Series([waktu[i] for i in m_])
            nep = aw.groupby([aw.dt.year, aw.dt.month]).size() if len(aw) else pd.Series(dtype=int)
            g = pd.concat([hari, terpicu], axis=1).reset_index()
            g["n_episode"] = [int(nep.get((r.thn, r.bln), 0)) for r in g.itertuples()]
            g["zona"], g["tier"], g["u_k"] = kunci, tier, round(u,4)
            t0 = waktu[0]
            g["t_tengah"] = [pd.Timestamp(y,m,15).dayofyear for y,m in zip(g.thn,g.bln)]
            g["t_absolut"] = [(pd.Timestamp(y,m,15)-t0).days for y,m in zip(g.thn,g.bln)]
            for j in (1,2):
                g[f"cos{j}"] = np.cos(2*np.pi*j*g.t_absolut/T).round(6)
                g[f"sin{j}"] = np.sin(2*np.pi*j*g.t_absolut/T).round(6)
            g["musim"] = [MUSIM[m] for m in g.bln]
            cac.append(g.rename(columns={"thn":"tahun","bln":"bulan"}))
    ep = pd.DataFrame(eps); ep.to_csv(AKAR/"data/processed/episode.csv", index=False)
    cb = pd.concat(cac, ignore_index=True)[["zona","tier","tahun","bulan","t_tengah","t_absolut",
        "hari","n_episode","n_hari_terpicu","cos1","sin1","cos2","sin2","musim","u_k"]]
    cb.to_csv(AKAR/"data/processed/cacah_bulanan.csv", index=False)

    assert harian.groupby("zona").tanggal.nunique().eq(len(waktu)).all(), "tanggal bolong/ganda"
    assert (ep.durasi == (pd.to_datetime(ep.tgl_akhir)-pd.to_datetime(ep.tgl_mulai)).dt.days+1).all(), "durasi tak konsisten"
    assert cb.groupby(["zona","tier"]).n_episode.sum().equals(ep.groupby(["zona","tier"]).size()), "cacah != jumlah episode"
    assert cb[["cos1","sin1","cos2","sin2"]].abs().le(1).all().all(), "regresor Fourier di luar [-1,1]"
    assert (cb.n_hari_terpicu >= 0).all() and (cb.hari > 0).all()

    n_thn = len(waktu)/365.25
    print(f"rekaman: {waktu[0].date()} s/d {waktu[-1].date()}  ({n_thn:.1f} tahun, {len(waktu)} hari)\n")
    print(f"{'zona':<8} {'tier':<5} {'u_k (m)':>8} {'episode':>8} {'/thn':>6} {'mu_D':>6} {'maks':>5} {'hari terpicu':>13}")
    print("-"*68)
    for (z,t), g in ep.groupby(["zona","tier"]):
        hp = cb[(cb.zona==z)&(cb.tier==t)].n_hari_terpicu.sum()
        print(f"{z:<8} {t:<5} {g.u_k.iloc[0]:>8.3f} {len(g):>8} {len(g)/n_thn:>6.1f} "
              f"{g.durasi.mean():>6.2f} {g.durasi.max():>5} {hp:>13}")
    print(f"\n-> swh_harian_zona.csv ({len(harian)}), episode.csv ({len(ep)}), cacah_bulanan.csv ({len(cb)})")
    print(f"-> dim_zona.csv (3 zona)")


if __name__ == "__main__":
    main()
