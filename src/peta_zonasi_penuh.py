"""fig 3.1, zone map"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from pathlib import Path
from global_land_mask import globe

AKAR = Path(__file__).resolve().parent.parent
PROCESSED, DIM, TAB, FIG = (AKAR / "data" / "processed", AKAR / "data" / "dim",
                            AKAR / "output" / "tables", AKAR / "output" / "figures")

PALET = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
DARAT, GARIS, TINTA, KABUR = "#ded9d1", "#8f8b83", "#2f2c28", "#7d7973"
LAUT_KOSONG = "#f4f6f7"
HALO = [pe.withStroke(linewidth=2.9, foreground="white")]

sel = pd.read_csv(PROCESSED / "zona_semua_sel_wpp573.csv")
pes = pd.read_csv(PROCESSED / "sel_zona_wpp573.csv")
dim = pd.read_csv(DIM / "dim_zona_wpp573.csv")
sik = pd.read_csv(TAB / "siklus_musiman_zona_wpp573.csv")
warna = {z: PALET[(z - 1) % len(PALET)] for z in sorted(dim.zona)}

WILAYAH = {
    1: "Pesisir selatan Jawa–Bali–Sumbawa + Samudra Hindia",
    2: "Selat Madura & utara Bali (Situbondo–Banyuwangi)",
    3: "Laut Flores, utara Sumbawa–Flores",
    4: "Sekitar Sumba & Laut Sawu, selatan Flores",
    5: "Tersebar — Laut Timor selatan & sel tepi kotak",
    6: "Pesisir Timor–Rote, Laut Timor",
}
BULAN_PANJANG = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
                 "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

LON0, LON1, LAT0, LAT1 = 103.8, 128.7, -15.6, -6.15
gl_lon = np.arange(LON0, LON1 + 1e-9, 0.02)
gl_lat = np.arange(LAT0, LAT1 + 1e-9, 0.02)
LO, LA = np.meshgrid(gl_lon, gl_lat)
darat = globe.is_land(LA, LO)

PULAU = [(110.6, -7.35, "J A W A", 11.5), (115.15, -8.32, "BALI", 7),
         (116.35, -8.55, "LOMBOK", 7), (117.9, -8.62, "SUMBAWA", 7.4),
         (121.3, -8.62, "FLORES", 7.4), (120.0, -9.85, "SUMBA", 7.4),
         (124.9, -9.45, "T I M O R", 8.4), (104.6, -6.95, "SUMATRA", 7.4)]
PELABUHAN = [
    (106.55, -7.02, "Palabuhanratu", .55, -.25, "right"),
    (108.65, -7.70, "Pangandaran", .82, -.55, "right"),
    (109.02, -7.73, "Cilacap", .30, .10, "center"),
    (110.33, -8.02, "Parangtritis", .80, .00, "center"),
    (111.73, -8.30, "Prigi", .74, .00, "center"),
    (113.47, -8.38, "Puger", .78, -.10, "center"),
    (114.37, -8.42, "Banyuwangi", .30, .35, "left")]

fig = plt.figure(figsize=(15.5, 14.2))
gs = fig.add_gridspec(3, 3, height_ratios=[2.75, 1.05, 1.05], hspace=.26, wspace=.09)

ax = fig.add_subplot(gs[0, :])
ax.set_facecolor(LAUT_KOSONG)

for _, r in sel.iterrows():
    ax.add_patch(Rectangle((r.lon - .25, r.lat - .25), .5, .5,
                           facecolor=warna[r.zona], edgecolor="none",
                           alpha=.80, zorder=2))

peta_zona = {(round(r.lon, 2), round(r.lat, 2)): r.zona for _, r in sel.iterrows()}
for (lo, la), z in peta_zona.items():
    for dlo, dla in ((.5, 0), (0, -.5)):
        t = peta_zona.get((round(lo + dlo, 2), round(la + dla, 2)))
        if t is not None and t != z:
            if dlo:
                ax.plot([lo + .25, lo + .25], [la - .25, la + .25],
                        color="white", linewidth=1.7, zorder=3)
            else:
                ax.plot([lo - .25, lo + .25], [la - .25, la - .25],
                        color="white", linewidth=1.7, zorder=3)

for _, r in pes.iterrows():
    ax.add_patch(Rectangle((r.lon - .25, r.lat - .25), .5, .5, facecolor="none",
                           edgecolor="white", linewidth=1.0, zorder=4))
    ax.add_patch(Rectangle((r.lon - .19, r.lat - .19), .38, .38,
                           facecolor=warna[r.zona], edgecolor="none", zorder=5))

ax.contourf(LO, LA, darat.astype(float), levels=[.5, 1.5], colors=[DARAT], zorder=6)
ax.contour(LO, LA, darat.astype(float), levels=[.5], colors=[GARIS],
           linewidths=.9, zorder=7)

for lon, lat, nama, uk in PULAU:
    ax.annotate(nama, (lon, lat), ha="center", va="center", fontsize=uk,
                color="#5f5a53", zorder=9, fontweight="bold", path_effects=HALO)
for lon, lat, nama, dy, dx, rata in PELABUHAN:
    ax.plot([lon, lon], [lat, lat + dy - .06], color=TINTA, linewidth=.6, zorder=9)
    ax.plot(lon, lat, "o", ms=3.4, color=TINTA, zorder=10,
            markeredgecolor="white", markeredgewidth=.7)
    ax.annotate(nama, (lon + dx, lat + dy), ha=rata, va="bottom", fontsize=7.6,
                color=TINTA, zorder=10, path_effects=HALO)

POSISI = {int(z): (g.lon.median(), g.lat.median()) for z, g in sel.groupby("zona")}
for _, r in dim.iterrows():
    z = int(r.zona); x, y = POSISI[z]
    ax.annotate(f"ZONA {z}", (x, y), ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=warna[z], zorder=11,
                path_effects=[pe.withStroke(linewidth=4.2, foreground="white")])

ax.annotate("petak besar  =  sel laut yang diwakili indeks zona itu\n"
            "petak kecil berbingkai  =  sel pesisir pembentuk indeks",
            (104.05, -14.55), fontsize=7.6, color="white", zorder=11,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="#00000055")])

ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1); ax.set_aspect(1.0)
ax.set_xticks(np.arange(104, 129, 2)); ax.set_yticks(np.arange(-15, -6, 1))
ax.set_xticklabels([f"{v:.0f}°BT" for v in np.arange(104, 129, 2)], fontsize=8.5)
ax.set_yticklabels([f"{abs(v):.0f}°LS" for v in np.arange(-15, -6, 1)], fontsize=8.5)
ax.grid(alpha=.14, linewidth=.6, zorder=1); ax.set_axisbelow(False)
ax.set_title("(a) Enam zona indeks WPP 573 — seluruh 776 sel laut diwarnai menurut "
             "zona yang indeksnya paling mewakilinya\n"
             "penugasan berdasarkan korelasi tertinggi deret SWH harian 2001–2025 "
             "terhadap indeks tiap zona, dihaluskan dengan modus tetangga",
             fontsize=11.5, loc="left", pad=10)

cakupan = sel.groupby("zona").agg(n_laut=("zona", "size"),
                                  kor_rata=("korelasi_thd_indeks", "mean"),
                                  kor_min=("korelasi_thd_indeks", "min"))
kartu_pertama = None
for i, (_, r) in enumerate(dim.sort_values("zona").iterrows()):
    z = int(r.zona)
    axc = fig.add_subplot(gs[1 + i // 3, i % 3])
    if kartu_pertama is None:
        kartu_pertama = axc
    axc.set_xlim(0, 1); axc.set_ylim(0, 1); axc.axis("off")
    c = warna[z]
    T = axc.transAxes

    axc.add_patch(FancyBboxPatch((.01, .015), .98, .97, boxstyle="round,pad=0.012",
                                 facecolor="#fcfbfa", edgecolor="#e0dcd6",
                                 linewidth=1, transform=T))
    axc.add_patch(Rectangle((.01, .875), .98, .11, facecolor=c, alpha=.93, transform=T))
    KUNCI = {1: "573-A", 4: "573-D", 6: "573-F"}
    judul = f"ZONA {z}  ·  {KUNCI[z]}" if z in KUNCI else f"ZONA {z}"
    axc.text(.04, .930, judul, fontsize=12.5, fontweight="bold", color="white",
             va="center", transform=T)
    axc.text(.96, .930, r.watak.upper(), fontsize=7.2, fontweight="bold", color="white",
             va="center", ha="right", transform=T)

    axc.text(.04, .835, WILAYAH[z].replace("\n", " "), fontsize=8.4, color=TINTA,
             va="top", transform=T, wrap=True)

    baris = [
        ("Sel pesisir → sel laut diwakili", f"{int(r.n_sel)} → {int(cakupan.loc[z, 'n_laut'])}"),
        ("Rentang bujur", f"{r.lon_min:.1f}–{r.lon_maks:.1f}°BT"),
        ("SWH rerata", f"{r.swh_rata:.2f} m"),
        ("Ambang pemicu (p90 lokal)", f"{r.ambang_m:.2f} m"),
        ("Korelasi internal antar sel", f"{r.korelasi_intra:.2f}"),
        ("Rasio musim barat : timur", f"{r.rasio_barat_timur:.2f}"),
    ]
    y = .745
    for k, v in baris:
        tebal = "bold" if k.startswith(("SWH", "Ambang")) else "normal"
        axc.text(.04, y, k, fontsize=7.7, color=KABUR, va="center", transform=T)
        axc.text(.63, y, v, fontsize=8.2, color=TINTA, va="center",
                 fontweight=tebal, transform=T)
        y -= .072

    axc.plot([.04, .96], [.335, .335], color="#e4e0da", linewidth=1, transform=T)

    s_z = sik[sik.zona == z].sort_values("bulan")
    pk = int(s_z.loc[s_z.swh_ternormalisasi.idxmax(), "bulan"])
    musim = "musim BARAT" if r.rasio_barat_timur > 1 else "musim TIMUR"
    axc.text(.04, .262, "Paling berbahaya saat", fontsize=7.5, color=KABUR,
             va="center", transform=T)
    axc.text(.04, .188, musim, fontsize=10.5, color=c, fontweight="bold",
             va="center", transform=T)
    axc.text(.04, .108, f"puncak bulan {BULAN_PANJANG[pk-1]}", fontsize=7.6,
             color=TINTA, va="center", transform=T)
    axc.text(.04, .046, f"korelasi sel–indeks {cakupan.loc[z, 'kor_rata']:.2f} "
                        f"(terendah {cakupan.loc[z, 'kor_min']:.2f})",
             fontsize=7.2, color=KABUR, va="center", transform=T)

    axs = axc.inset_axes([.575, .055, .39, .195])
    axs.plot(s_z.bulan, s_z.swh_ternormalisasi, "-", color=c, linewidth=1.9)
    axs.axhline(1.0, color="#d5d1ca", linewidth=.8)
    axs.plot(pk, s_z.swh_ternormalisasi.max(), "o", ms=4.5, color=c)
    axs.set_xlim(.5, 12.5); axs.set_ylim(.55, 2.0)
    axs.set_xticks([1, 4, 7, 10]); axs.set_xticklabels(["J", "A", "J", "O"], fontsize=6)
    axs.set_yticks([]); axs.tick_params(length=1.5, pad=1)
    for sp in axs.spines.values():
        sp.set_visible(False)
    axs.set_facecolor("none")
    axc.text(.575, .272, "siklus musiman", fontsize=7.2, color=KABUR, transform=T)

pos = kartu_pertama.get_position()
fig.text(pos.x0, pos.y1 + .018,
         "(b) Karakteristik tiap zona   —   ambang pemicu disetarakan pada "
         "36,5 hari terpicu per tahun, sehingga yang berbeda adalah TINGGI ambangnya, "
         "bukan frekuensinya",
         fontsize=11.5, ha="left", color=TINTA)

FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "fig_zonasi_peta_penuh.png", dpi=170, bbox_inches="tight",
            facecolor="white")
print("selesai:", FIG / "fig_zonasi_peta_penuh.png")
