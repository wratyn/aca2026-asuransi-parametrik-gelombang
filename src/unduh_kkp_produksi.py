#!/usr/bin/env python3
"""pull KKP marine capture data from the open API"""
import argparse
import csv
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import json

AKAR = Path(__file__).resolve().parent.parent
KELUAR = AKAR / "data" / "raw" / "kkp_produksi"

BASIS = "https://portaldata.kkp.go.id/sttk/api/statistik/list/perikanan_tangkap_laut_kab_kota"
KOLOM = ("tahun,provinsi,kabupaten_kota,lembar_kerja,pelabuhan,wpp,"
         "jenis_kapal,jenis_api,kelompok,jenis_ikan")
JEDA = 1.5
PERCOBAAN = 4


def ambil(page, limit, timeout=90):
    url = BASIS + "?" + urlencode({
        "page": page, "limit": limit, "kolom": KOLOM,
        "sorto": "DESC", "sortc": "tahun",
    })
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (riset akademik ACA2026)"})
    galat = None
    for k in range(PERCOBAAN):
        try:
            with urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as e:
            galat = e
            tunggu = 5 * (k + 1)
            print(f"    gagal ({e.__class__.__name__}), coba lagi dalam {tunggu}s", flush=True)
            time.sleep(tunggu)
    raise RuntimeError(f"halaman {page} gagal setelah {PERCOBAAN} percobaan: {galat}")


def unduh_semua(limit, mulai_hal):
    KELUAR.mkdir(parents=True, exist_ok=True)
    penulis, berkas_buka, cacah = {}, {}, {}
    page, total = mulai_hal, None
    try:
        while True:
            hasil = ambil(page, limit)
            data = hasil.get("data", [])
            if total is None:
                total = int(hasil.get("total_count", 0) or 0)
                print(f"total_count = {total:,} baris, limit={limit} "
                      f"-> ~{-(-total // limit)} halaman", flush=True)
            if not data:
                break

            for d in data:
                th = str(d.get("tahun", "")).strip()
                if not th:
                    continue
                if th not in penulis:
                    b = KELUAR / f"produksi_tangkap_laut_{th}.csv"
                    f = b.open("w", newline="", encoding="utf-8")
                    berkas_buka[th] = f
                    penulis[th] = csv.DictWriter(f, fieldnames=list(d.keys()))
                    penulis[th].writeheader()
                    cacah[th] = 0
                penulis[th].writerow(d)
                cacah[th] += 1

            ringkas = " ".join(f"{k}:{v:,}" for k, v in sorted(cacah.items()))
            print(f"hal {page} (+{len(data)})  |  {ringkas}", flush=True)
            page += 1
            time.sleep(JEDA)
    finally:
        for f in berkas_buka.values():
            f.close()

    print("\nselesai. baris per tahun:")
    for k, v in sorted(cacah.items()):
        print(f"  {k}: {v:,}")
    print(f"keluaran -> {KELUAR}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=5000,
                   help="baris per halaman; turunkan kalau sering timeout")
    p.add_argument("--mulai-hal", type=int, default=1,
                   help="lanjutkan dari halaman ini kalau tarikan sempat putus")
    a = p.parse_args()
    try:
        unduh_semua(a.limit, a.mulai_hal)
    except Exception as e:
        print(f"GAGAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
