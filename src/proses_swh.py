#!/usr/bin/env python3
"""ERA5 .nc -> daily SWH csv"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swh_core as core

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"


def cari_file(wpp, pola, awal=None, akhir=None):
    pat = pola if pola else f"*wpp{wpp}*.nc"
    files = sorted(RAW.glob(pat))
    if not files:
        raise SystemExit(f"Tidak ada file cocok '{pat}' di {RAW}\n"
                         f"Jalankan dulu: python src/download_era5_swh.py --wpp {wpp}")

    if awal is None and akhir is None:
        return files, []

    dipakai, dilewati = [], []
    for f in files:
        th = core.tahun_dari_nama(f)
        if th is None:
            dipakai.append(f)
        elif (awal is not None and th < awal) or (akhir is not None and th > akhir):
            dilewati.append(f)
        else:
            dipakai.append(f)
    if not dipakai:
        raise SystemExit(f"Semua file di luar jendela {awal}-{akhir}. "
                         f"Unduh dulu tahun yang diminta.")
    return dipakai, dilewati


def plot_deret(df, wpp, outdir, sfx=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(df["tanggal"], df["swh_mean"], lw=0.8, color="#1f4e79", label="indeks (rata-rata WPP)")
    ax.fill_between(df["tanggal"], df["swh_mean"], df["swh_max"],
                    color="#1f4e79", alpha=0.12, lw=0, label="s/d maks antar-grid")
    for nama, a in core.AMBANG_TIER.items():
        ax.axhline(a, ls="--", lw=0.8, color="#b00020", alpha=0.7)
        ax.annotate(f"{nama} ({a} m)", xy=(1.002, a), xycoords=("axes fraction", "data"),
                    fontsize=7, va="center", color="#b00020")
    ax.set_ylabel("SWH maksimum harian (m)")
    ax.set_xlabel("")
    ax.set_title(f"Gambar 3.1 — SWH harian WPP {wpp} ({core.WPP_NAMA.get(wpp, '')})", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.margins(x=0.01)
    fig.tight_layout()
    out = outdir / f"fig_3_1_deret_swh_wpp{wpp}{sfx}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_siklus(cacah, wpp, ambang, outdir, sfx=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rata = cacah.groupby("bulan")["n_terpicu"].mean()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(rata.index, rata.values, color="#1f4e79", width=0.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                        "Jul", "Agu", "Sep", "Okt", "Nov", "Des"], fontsize=8)
    ax.set_ylabel(f"Rata-rata hari SWH > {ambang} m")
    ax.set_title(f"Gambar 3.2 — Pola musiman frekuensi pemicu, WPP {wpp}", fontsize=10)
    fig.tight_layout()
    out = outdir / f"fig_3_2_siklus_bulanan_wpp{wpp}{sfx}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wpp", default="573", choices=sorted(core.WPP_BBOX))
    ap.add_argument("--pola", default=None, help="pola glob di data/raw, mis. 'TEST_*.nc'")
    ap.add_argument("--start", type=int, default=None, help="tahun awal jendela kalibrasi")
    ap.add_argument("--end", type=int, default=None, help="tahun akhir jendela kalibrasi")
    ap.add_argument("--ambang", type=float, default=2.0, help="ambang untuk cacah bulanan (m)")
    ap.add_argument("--suffix", default="", help="akhiran nama file output, mis. '_TEST'")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    for d in (PROCESSED, TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)

    files, dilewati = cari_file(args.wpp, args.pola, args.start, args.end)
    print(f"WPP {args.wpp} — {core.WPP_NAMA.get(args.wpp, '')}")
    print(f"File mentah ({len(files)} dipakai"
          + (f", {len(dilewati)} di luar jendela {args.start}-{args.end}" if dilewati else "")
          + "):")
    for f in files:
        print(f"  - {f.name}  ({f.stat().st_size/1024:.0f} KB)")

    df = core.build_series(files)
    sfx = args.suffix
    out_harian = PROCESSED / f"swh_harian_wpp{args.wpp}{sfx}.csv"
    df.to_csv(out_harian, index=False, float_format="%.4f")

    n_lengkap = int((df.groupby("tahun").size() >= 365).sum())
    print(f"\n[1] Deret harian -> {out_harian.name}")
    print(f"    {len(df):,} hari, {df.tanggal.min().date()} s/d {df.tanggal.max().date()}, "
          f"{df.n_grid.iloc[0]} sel laut per hari")
    print(f"    {df.tahun.nunique()} tahun tersentuh, {n_lengkap} di antaranya lengkap "
          f"(>=365 hari) — yang dipakai model siklis")
    if 0 < n_lengkap < core.N_TAHUN_DEFAULT:
        print(f"    ! baru {n_lengkap} tahun lengkap; galat premi murni masih besar. "
              f"Target minimal {core.N_TAHUN_DEFAULT}.")
    print(df[["swh_mean", "swh_p90", "swh_max"]].describe().round(3).to_string())

    cacah = core.cacah_bulanan(df, args.ambang)
    out_cacah = PROCESSED / f"cacah_bulanan_wpp{args.wpp}{sfx}.csv"
    cacah.to_csv(out_cacah, index=False, float_format="%.4f")
    print(f"\n[2] Cacah bulanan (ambang {args.ambang} m) -> {out_cacah.name}")
    print(cacah.head(12).to_string(index=False))

    tab = core.ringkas_ambang(df)
    out_tab = TABLES / f"ambang_vs_frekuensi_wpp{args.wpp}{sfx}.csv"
    tab.to_csv(out_tab, index=False, float_format="%.3f")
    print(f"\n[3] Kandidat ambang -> {out_tab.name}")
    print(tab.round(2).to_string(index=False))

    if not args.no_plot:
        print()
        print("[4] Gambar ->", plot_deret(df, args.wpp, FIGURES, sfx).name)
        if cacah["tahun"].nunique() >= 2 or cacah["bulan"].nunique() >= 6:
            print("            ", plot_siklus(cacah, args.wpp, args.ambang, FIGURES, sfx).name)
        else:
            print("             (fig 3.2 dilewati — data belum cukup panjang untuk pola musiman)")


if __name__ == "__main__":
    main()
