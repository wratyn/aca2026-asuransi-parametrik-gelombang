#!/usr/bin/env python3
"""fig 3.2, one median season"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import pandas as pd, numpy as np
from pathlib import Path
AKAR=Path(__file__).resolve().parent.parent
TINTA="#2b2b2b"; KABUR="#8a8a8a"; BIRU="#1f77b4"; MERAH="#d1495b"; HIJAU="#2a9d8f"

h=pd.read_csv(AKAR/"data/processed/swh_harian_zona.csv"); h["tanggal"]=pd.to_datetime(h.tanggal)
dt=pd.read_csv(AKAR/"data/dim/dim_tier.csv")
u1=float(dt[(dt.zona=="573-A")&(dt.tier==1)].u_k_awal.iloc[0])
ep=pd.read_csv(AKAR/"data/processed/episode.csv")
e=ep[(ep.zona=="573-A")&(ep.tier==1)].copy()
e["tgl_mulai"]=pd.to_datetime(e.tgl_mulai); e["tgl_akhir"]=pd.to_datetime(e.tgl_akhir)

M0,M1=pd.Timestamp("2023-06-01"),pd.Timestamp("2023-08-31")
s=h[(h.zona=="573-A")&(h.tanggal.between(M0,M1))].sort_values("tanggal")
es=e[(e.tgl_mulai>=M0)&(e.tgl_mulai<=M1)].sort_values("tgl_mulai")

fig=plt.figure(figsize=(13.5,7.6)); fig.patch.set_facecolor("white")
gs=fig.add_gridspec(2,1,height_ratios=[2.5,1],hspace=.32,left=.07,right=.98,top=.87,bottom=.09)

ax=fig.add_subplot(gs[0])
ax.plot(s.tanggal,s.swh_rerata,color=BIRU,lw=1.6,zorder=3)
ax.axhline(u1,color=MERAH,lw=1.4,ls="--",zorder=4)
ax.text(M0,u1+.03,f"  ambang pemicu tier 1  u = {u1:.2f} m",color=MERAH,fontsize=9,va="bottom",fontweight="bold")
for _,r in es.iterrows():
    ax.axvspan(r.tgl_mulai-pd.Timedelta(hours=12),r.tgl_akhir+pd.Timedelta(hours=12),
               color=MERAH,alpha=.13,zorder=1)
    ax.annotate(f"{r.durasi}",(r.tgl_mulai+(r.tgl_akhir-r.tgl_mulai)/2,ax.get_ylim()[1]),
                ha="center",va="top",fontsize=8,color=MERAH,fontweight="bold")
ax.fill_between(s.tanggal,u1,s.swh_rerata,where=s.swh_rerata>u1,color=MERAH,alpha=.35,zorder=2)
ax.set_ylabel("tinggi gelombang indeks zona 573-A  (m)",fontsize=9.5,color=TINTA)
ax.set_title("Musim Timur 2023 — musim dengan jumlah hari terpicu persis MEDIAN dari 25 musim (28 hari)",
             fontsize=11,color=TINTA,loc="left",pad=26,fontweight="bold")
ax.text(0,1.045,"Tiap blok merah = satu episode badai. Angka di atasnya = jumlah hari dibayar.",
        transform=ax.transAxes,fontsize=9,color=KABUR)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8.5,colors=KABUR); ax.set_xlim(M0,M1)

ax2=fig.add_subplot(gs[1])
kum=(s.swh_rerata>u1).cumsum()
ax2.step(s.tanggal,kum,color=HIJAU,lw=1.9,where="post",zorder=3)
for f,lab,c in ((18,"minimum 25 musim: 18 hari","#b0b0b0"),(28,"median: 28 hari",KABUR)):
    ax2.axhline(f,color=c,lw=1,ls=":",zorder=2)
    ax2.text(M1,f,f" {lab}",fontsize=8,color=c,va="center")
ax2.set_ylabel("kumulatif hari dibayar",fontsize=9.5,color=TINTA)
ax2.set_xlim(M0,M1); ax2.set_ylim(0,34)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
for sp in ("top","right"): ax2.spines[sp].set_visible(False)
ax2.tick_params(labelsize=8.5,colors=KABUR)
ax2.text(0,-.42,"Selama 25 musim timur, jumlah hari terpicu tidak pernah di bawah 18 dan tidak pernah nol.\n"
        "Artinya 62% payout tier 1 sebenarnya sudah pasti — bagian itu bukan asuransi, melainkan transfer terjadwal.",
        transform=ax2.transAxes,fontsize=9,color=MERAH,va="top")
fig.savefig(AKAR/"output/figures/fig_mekanisme_musim.png",dpi=150,facecolor="white",bbox_inches="tight")
print("selesai:",AKAR/"output/figures/fig_mekanisme_musim.png")
