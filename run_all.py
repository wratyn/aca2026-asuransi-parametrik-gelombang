#!/usr/bin/env python3
"""rerun everything. flags: --tanpa-notebook, --dari ID, --daftar, --cek"""
import argparse, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

AKAR = Path(__file__).resolve().parent
PY = sys.executable

LANGKAH = [
    ("00",  "0 Seleksi WPP",        ["src/00_seleksi_wpp.py"], "composite scoring 11 WPP (butuh indikator_wpp.csv)"),
    ("01",  "1 Data ERA5",          ["src/cek_kelengkapan.py", "--wpp", "573", "--start", "2001", "--end", "2025"],
                                    "cek 25 berkas ERA5 lengkap & konsisten"),
    ("02",  "2 Eksplorasi awal",    ["src/proses_swh.py", "--wpp", "573", "--start", "2021", "--end", "2025"],
                                    "deret SWH harian WPP 573 + kandidat ambang (5 tahun)"),
    ("03",  "2 Eksplorasi awal",    ["src/analisis_basis_risk.py"], "adu rancangan indeks A/B/C (ERA5 2021-2024)"),
    ("04",  "2 Eksplorasi awal",    ["src/durasi_dan_gambar.py"], "durasi episode + gambar diagnostik (2021-2024)"),
    ("nb1", "2 Eksplorasi awal",    ["notebooks/01_eksplorasi_swh.ipynb"], "notebook eksplorasi ERA5"),
    ("nb2", "2 Eksplorasi awal",    ["notebooks/02_analisis_swh_5tahun.ipynb"], "notebook analisis 5 tahun"),
    ("nb3", "2 Eksplorasi awal",    ["notebooks/03_eda_swh_5tahun.ipynb"], "notebook EDA 5 tahun"),
    ("Z1",  "3 Zonasi",             ["src/zonasi.py"], "zonasi berbasis korelasi, 25 tahun -> k = 6"),
    ("Z2",  "3 Zonasi",             ["src/zonasi_semua_sel.py"], "penugasan seluruh 776 sel laut ke zona"),
    ("Z3",  "3 Zonasi",             ["src/peta_zonasi_penuh.py"], "Gambar 3.1 makalah"),
    ("nb4", "3 Zonasi",             ["notebooks/04_zonasi_berbasis_data.ipynb"], "notebook zonasi"),
    ("05",  "4 Data sosial-ekonomi", ["src/05_eksposur_dan_bridge.py"], "eksposur kapal & bobot provinsi->zona"),
    ("06",  "4 Data sosial-ekonomi", ["src/06_panel_ntn.py"], "panel NTN 8 provinsi"),
    ("10",  "5 Indeks & frekuensi", ["src/10_pipeline_indeks.py"], "indeks zona, episode, cacah bulanan"),
    ("10b", "5 Indeks & frekuensi", ["src/10b_dim_tier_musim.py"], "dimensi tier & musim"),
    ("11",  "5 Indeks & frekuensi", ["src/11_fit_lambda.py"], "GLM Poisson siklis (Fourier)"),
    ("12",  "5 Indeks & frekuensi", ["src/12_basis_risk.py"], "uji proksi NTN (ditolak)"),
    ("13",  "5 Indeks & frekuensi", ["src/13_basis_risk_spasial.py"], "basis risk spasial"),
    ("14",  "5 Indeks & frekuensi", ["src/14_gambar_mekanisme.py"], "Gambar 3.2 makalah"),
    ("15",  "6 Desain polis",       ["src/15_kalibrasi_bk.py"], "manfaat harian b_k dari KKP + literatur"),
    ("16",  "6 Desain polis",       ["src/16_desain_franchise.py"], "kandidat franchise"),
    ("17",  "6 Desain polis",       ["src/17_optimasi_uk_franchise.py"], "optimasi bersama u_k & franchise"),
    ("18",  "6 Desain polis",       ["src/18_kunci_desain.py"], "kunci desain 20 Sep"),
    ("11b", "6 Desain polis",       ["src/11b_gambar_intensitas.py"], "Gambar 3.3 makalah"),
    ("19",  "7 Harga & modal",      ["src/19_simulasi_skenario.py"], "adu 10 skenario rancangan"),
    ("20",  "7 Harga & modal",      ["src/20_fit_durasi.py"], "distribusi durasi + theta Ferro-Segers"),
    ("21",  "7 Harga & modal",      ["src/21_panjer.py"], "rekursi Panjer (Poisson)"),
    ("22",  "7 Harga & modal",      ["src/22_panjer_ab0.py"], "kelas (a,b,0): binomial"),
    ("23",  "7 Harga & modal",      ["src/23_premi_dan_modal.py"], "premi bruto, TVaR, RBC"),
    ("24",  "7 Harga & modal",      ["src/24_sensitivitas.py"], "sensitivitas satu-faktor"),
    ("25",  "8 Dampak & perluasan", ["src/25_dampak_nelayan.py"], "dampak ke nelayan (21 Sep)"),
    ("26",  "8 Dampak & perluasan", ["src/26_lapis_jiwa.py"], "rider jiwa multi-state"),
    ("27",  "8 Dampak & perluasan", ["src/27_reasuransi.py"], "struktur reasuransi"),
    ("28",  "8 Dampak & perluasan", ["src/28_dampak_skenario_24sep.py"], "dampak skenario final 24 Sep"),
    ("V",   "9 Verifikasi",         ["src/verifikasi.py"], "verifikasi independen angka kunci"),
]

DIPANTAU = ["data/dim", "data/processed", "output/tables"]


def jalankan(cmd, log):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
    if cmd[0].endswith(".ipynb"):
        nb = AKAR / cmd[0]
        with tempfile.TemporaryDirectory() as tmp:
            args = [PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--ExecutePreprocessor.timeout=1200", "--output-dir", tmp, str(nb)]
            return subprocess.run(args, cwd=nb.parent, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return subprocess.run([PY, *cmd], cwd=AKAR, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tanpa-notebook", action="store_true")
    ap.add_argument("--dari", default=None, help="ID langkah awal")
    ap.add_argument("--daftar", action="store_true")
    ap.add_argument("--cek", action="store_true", help="bandingkan hasil baru dengan hasil yang ada")
    a = ap.parse_args()

    langkah = [l for l in LANGKAH if not (a.tanpa_notebook and l[2][0].endswith(".ipynb"))]
    if a.dari:
        ids = [l[0] for l in langkah]
        if a.dari not in ids:
            sys.exit(f"ID {a.dari} tidak dikenal. Pilihan: {', '.join(ids)}")
        langkah = langkah[ids.index(a.dari):]
    if a.daftar:
        for i, t, c, k in langkah:
            print(f"{i:>4}  {t:<22} {' '.join(c):<48} {k}")
        return

    ref = None
    if a.cek:
        ref = Path(tempfile.mkdtemp(prefix="aca_ref_"))
        for d in DIPANTAU:
            if (AKAR / d).exists():
                shutil.copytree(AKAR / d, ref / d)
        print(f"hasil lama disalin ke {ref}\n")

    for d in ("data/dim", "data/processed", "output/tables", "output/figures", "log"):
        (AKAR / d).mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for i, tahap, cmd, ket in langkah:
        t0 = time.time()
        with open(AKAR / "log" / f"{i}.log", "w") as log:
            rc = jalankan(cmd, log)
        status = "ok" if rc == 0 else f"GAGAL (kode {rc}) — lihat log/{i}.log"
        print(f"[{i:>3}] {tahap:<22} {ket:<50} {time.time()-t0:6.1f} dtk  {status}")
        if rc != 0:
            sys.exit(1)
    print(f"\nselesai dalam {time.time()-t_all:.0f} detik. Log per langkah di log/.")

    if ref is not None:
        sys.path.insert(0, str(AKAR / "tools"))
        from cek_reproduksi import bandingkan_folder
        ok = bandingkan_folder(ref, AKAR, DIPANTAU)
        sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
