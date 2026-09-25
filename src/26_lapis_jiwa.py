#!/usr/bin/env python3
"""life/disability rider, multi-state"""
from pathlib import Path
import numpy as np, pandas as pd
AKAR = Path(__file__).resolve().parent.parent

PREMI_BPAN, B03, B02, B_MED = 175_000, 200e6, 100e6, 20e6
RASIO_MURNI = {"longgar 0,60": 0.60, "tengah 0,70": 0.70, "ketat 0,80": 0.80}
CAMPURAN = {"hanya kematian": (0.0, 0.0), "cacat 0,5q + medis 5q": (0.5, 5.0)}
P_U = {1: 0.15, 2: 0.10}
ETA  = {"eta=ln2": np.log(2), "eta=ln3": np.log(3), "eta=ln5": np.log(5)}
PSI  = [0.2, 0.4, 0.6]
N_T1, N_T2 = 48185, 5768
ABK = {1: 2.0, 2: 5.0}

rows = []
for nama_r, r in RASIO_MURNI.items():
    murni = PREMI_BPAN*r
    for nama_c, (a_cacat, a_med) in CAMPURAN.items():
        q = murni/(B03 + a_cacat*B02 + a_med*B_MED)
        rows.append(dict(blok="kalibrasi", rasio_murni=nama_r, campuran=nama_c,
                         premi_murni=murni, q03_implisit=q, per_100rb=1e5*q))
K = pd.DataFrame(rows)
q_pakai = float(K[(K.rasio_murni=="tengah 0,70")&(K.campuran=="cacat 0,5q + medis 5q")].q03_implisit.iloc[0])

prot = []
for tier, p_u in P_U.items():
    for nama_e, eta in ETA.items():
        bobot = (1-p_u) + p_u*np.exp(eta)
        mu0 = q_pakai/bobot
        pangsa_hari_buruk = p_u*np.exp(eta)/bobot
        for psi in PSI:
            bobot_baru = (1-p_u) + p_u*(1-psi)*np.exp(eta)
            q_baru = mu0*bobot_baru
            n_jiwa = (N_T1 if tier==1 else N_T2)*ABK[tier]
            prot.append(dict(blok="proteksi", tier=tier, eta=nama_e, psi=psi,
                             q_awal=q_pakai, q_dengan_asuransi=q_baru,
                             turun_pct=100*(q_baru/q_pakai-1),
                             pangsa_kematian_hari_buruk=pangsa_hari_buruk,
                             jiwa=n_jiwa, kematian_awal=n_jiwa*q_pakai,
                             kematian_dicegah=n_jiwa*(q_pakai-q_baru)))
P = pd.DataFrame(prot)

rider = []
for B in (25e6, 50e6, 100e6):
    for tier in (1,2):
        murni = q_pakai*B + 0.5*q_pakai*(B/2) + 5*q_pakai*2e6
        rider.append(dict(blok="rider", tier=tier, santunan_kematian=B,
                          premi_murni_rider=murni, bruto_A=murni/(1-0.20)*1.0,
                          pct_dari_premi_indeks=100*(murni/(1-0.20))/ (6035772 if tier==1 else 15804494)))
R = pd.DataFrame(rider)
pd.concat([K, P, R]).to_csv(AKAR/"output/tables/lapis_jiwa.csv", index=False)

pd.set_option("display.width", 260)
print("== kalibrasi q03 implisit dari tarif BPAN ==")
print(K.round(6).to_string(index=False))
print("\nPembanding literatur: 103-129 kematian per 100.000 nelayan-tahun (UK, AS, Denmark).")
print("\n== efek protektif (q dipakai = %.2e, yaitu %.0f per 100.000) ==" % (q_pakai, 1e5*q_pakai))
print(P[P.tier==1].round(4).to_string(index=False))
print("\n== premi rider santunan (murni + biaya 20%) ==")
print(R.round(0).to_string(index=False))
