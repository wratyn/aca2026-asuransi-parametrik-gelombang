"""daily benefit b_k = R/D * m"""
import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
lit = pd.read_csv(ROOT/'data/raw/literatur/rekap_literatur_bk.csv')
bridge = pd.read_csv(ROOT/'data/dim/bridge_provinsi_zona.csv'); bridge = bridge[bridge.wpp=='WPP-RI-573'].copy()
eks = pd.read_csv(ROOT/'data/processed/eksposur_provinsi_gt.csv')

def tier(k):
    if k.startswith('MT_') or k == 'KM_0005': return 1
    if k in ('KM_0005_0010','KM_0010_0020','KM_0020_0030'): return 2
    if k.startswith('KM_0'): return 3
    return 0

rows = []
for y in (2022, 2023, 2024):
    p = pd.read_csv(ROOT/f'data/raw/kkp_produksi/produksi_tangkap_laut_{y}.csv', low_memory=False)
    p = p[p.wpp=='WPP-RI-573'].copy()
    p['nilai'] = pd.to_numeric(p.nilai_produksi_rp, errors='coerce')*1000  # KKP value is in thousand rupiah
    p['tier'] = p.jenis_kapal.map(tier)
    g = p[p.tier>0].groupby(['provinsi','tier']).nilai.sum().reset_index(); g['tahun'] = y
    rows.append(g)
v = pd.concat(rows)
bridge['s'] = bridge.bobot/bridge.groupby('provinsi').bobot.transform('sum')
bridge['zona_bk'] = bridge.zona.replace({'573-D':'573-DF','573-F':'573-DF'})
m = v.merge(bridge[['provinsi','zona_bk','bobot','s']], on='provinsi')
m = m.merge(eks[['provinsi','tier','n_2022','n_2023','n_2024']], on=['provinsi','tier'])
m['nilai_z'] = m.nilai*m.s
m['n'] = m.apply(lambda r: r[f'n_{r.tahun}'], axis=1)*m.bobot
a = m.groupby(['zona_bk','tier','tahun'])[['nilai_z','n']].sum().reset_index()
a['R'] = a.nilai_z/a.n
a = a[~((a.zona_bk=='573-DF') & (a.tahun==2024))]  # NTT 2024 looks like a data artifact
a = a[a.n >= 50]
R = a.groupby(['zona_bk','tier']).R.agg(R_med='median', R_min='min', R_max='max', n_tahun='count').reset_index()

d = lit[lit.dipakai_D==1].copy()
d['studi'] = d.sumber.str.split('—').str[0].str.strip() + d.tier.astype(str)
d = d.groupby(['tier','studi']).D_hari_tahun.mean().reset_index()
D = d.groupby('tier').D_hari_tahun.agg(D_base='median', D_min='min', D_max='max', D_nstudi='count').reset_index()
tunggal = D.D_min == D.D_max
D.loc[tunggal & (D.tier==2), ['D_min','D_max']] = [D.loc[D.tier==2,'D_base'].iloc[0]*0.75, D.loc[D.tier==2,'D_base'].iloc[0]*1.25]
D.loc[D.tier==3, ['D_min','D_max']] = [180, 360]

mm = lit[lit.dipakai_m==1]
M = mm.groupby('tier').agg(m_base=('m_kontribusi','median'), m_min=('m_kontribusi','min'),
                           m_max=('m_kontribusi','max'), m_bersih=('m_bersih','median'),
                           m_nstudi=('m_kontribusi','count')).reset_index()
M.loc[M.tier==3, 'm_min'] = lit[lit.id.isin(['L13','L14','L15'])].m_bersih.median()
M.loc[M.tier==3, 'm_bersih'] = M.loc[M.tier==3, 'm_min']

out = R.merge(D, on='tier').merge(M, on='tier')
out['b_base']   = out.R_med/out.D_base*out.m_base
out['b_bersih'] = out.R_med/out.D_base*out.m_bersih
out['b_rendah'] = out.R_med/out.D_max*out.m_min
out['b_tinggi'] = out.R_med/out.D_min*out.m_max
for c in ['b_base','b_bersih','b_rendah','b_tinggi']:
    out[c] = (out[c]/1000).round()*1000
out = out.rename(columns={'zona_bk':'zona'})
out['basis_margin'] = 'bersih'; out['b_pakai'] = out.b_bersih
out.to_csv(ROOT/'output/tables/b_k_usulan.csv', index=False)
pd.set_option('display.width', 250)
print(a.round(0).to_string()); print(out.round(3).to_string())
