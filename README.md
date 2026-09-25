# aca2026-asuransi-parametrik-gelombang

Code and data behind our A-CA 2026 (ASSA IPB) paper: parametric income insurance for small-scale fishers in WPP 573, triggered by daily ERA5 significant wave height per zone.

## Run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Takes about 2 minutes. `python run_all.py --cek` reruns everything and diffs the tables against what's committed. `--daftar` lists the steps, `--dari <id>` starts from one, `--tanpa-notebook` skips the notebooks.

## Layout

```
src/                         scripts, numbered roughly in the order we did them
notebooks/                   early exploration on ERA5 2021–2025
data/raw/                    inputs, never edited
data/dim, data/processed     generated
output/tables, output/figures
tools/cek_reproduksi.py      table diff with numeric tolerance
```

## Steps

| step | scripts | |
|---|---|---|
| 0 | `00_seleksi_wpp` | WPP selection, composite score ([docs/pemilihan-wpp.pdf](docs/pemilihan-wpp.pdf)) |
| 1–2 | `download_era5_swh`, `proses_swh`, `analisis_basis_risk`, `durasi_dan_gambar` | ERA5 pull, first basis-risk test (absolute vs relative trigger) |
| 3 | `zonasi`, `zonasi_semua_sel`, `peta_zonasi_penuh` | correlation-based zoning with adjacency constraint, k = 6 |
| 4 | `05`, `06`, `unduh_kkp_produksi` | fleet exposure, province→zone weights, NTN panel |
| 5 | `10`–`14` | zone index, episodes, cyclic Poisson GLM, spatial basis risk |
| 6 | `15`–`18`, `11b` | daily benefit b_k, franchise, trigger optimisation, design lock |
| 7 | `19`–`24` | scenarios, Panjer with binomial counts, premium, TVaR capital, sensitivity |
| 8 | `25`–`28` | impact on fishers, life rider, reinsurance layers, final scenario |

Paper figures 3.1–3.3: `fig_zonasi_peta_penuh.png`, `fig_mekanisme_musim.png`, `fig_intensitas_final.png`.

## Data

| file | source |
|---|---|
| `era5_swh_wpp573_<yr>.nc` 2001–2025 | ERA5 daily max SWH, 0.5°, UTC+7 (Copernicus CDS) |
| `kkp_produksi/` 2019–2024 | Portal Data KKP, marine capture by regency × WPP (`nilai_produksi_rp` is in thousand rupiah) |
| `kkp_kapal/` | Portal Data KKP, vessel counts by province and GT class |
| `bps_ntn/` | BPS provincial fishers' terms of trade, monthly |
| `pipp/` | PIPP monthly landings, PPN Palabuhanratu & PPN Prigi |
| `literatur/` | fishing days, revenue and margins from 17 studies (URLs inside) |

To re-download: ERA5 with `src/download_era5_swh.py` (needs `~/.cdsapirc`), KKP production with `src/unduh_kkp_produksi.py`. Vessel counts and NTN were pulled by hand from the portals.

## Notes

- `data/raw/seleksi_wpp/indikator_wpp.csv` isn't in the repo yet, so step 00 just skips.
- The early exploration scripts default to ERA5 2021–2024; set `ACA_TAHUN_AWAL` / `ACA_TAHUN_AKHIR` to change it.
- `eksposur.csv` still carries the old zone labels (573-B, 573-E); downstream scripts map them to 573-A and 573-D/F.

## License

Code: MIT. Data follows its sources. ERA5: contains modified Copernicus Climate Change Service information (2026).
