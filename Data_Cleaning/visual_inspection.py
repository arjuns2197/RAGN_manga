
import cProfile, pstats, io
from pstats import SortKey
import sys
sys.path.append("..")
from AGN_FIRST_code import agn_functions as agn


import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from sklearn.linear_model import LinearRegression
import scipy.integrate as integrate
import pandas as pd
import scipy
from scipy.stats import norm
from scipy import ndimage
from scipy import interpolate
import cProfile
from astropy.table import Column
from astropy.coordinates import Angle
import matplotlib.patches as patches
import pylab as pl


df = pd.read_pickle("../AGN_FIRST_files/data/df_drpmatched_preVI.pkl")
inds = []

for i in range(10231):

    if type(df['Radio Class'][i]) == str:

        inds.append(i)


inds = np.array(inds)


num = 7

ind  = inds[num - 1]

ras_nvss = df['RA NVSS(3m)'][ind]
decs_nvss = df['DEC NVSS(3m)'][ind]
nvss_fluxes = np.log10(df['Flux NVSS(3m)'][ind])

ra_manga, dec_manga = df['RA'][ind], df['DEC'][ind]

ras_first = df['RA FIRST(30s)'][ind]
decs_first = df['DEC FIRST(30s)'][ind]


print(df['RA'][ind], df['DEC'][ind])

if ras_first is not None:
    first_fluxes = np.log10(df['Flux FIRST(30s)'][ind])

delta_ras_nvss = -np.cos(np.deg2rad(decs_nvss))*(ras_nvss - ra_manga)*3600
delta_decs_nvss = (decs_nvss - dec_manga)*3600

if ras_first is not None:
    delta_ras_first = -np.cos(np.deg2rad(decs_first))*(ras_first - ra_manga)*3600
    delta_decs_first = (decs_first - dec_manga)*3600

plt.scatter(delta_ras_nvss,
delta_decs_nvss, s = 200, label = 'NVSS Sources',marker = 'H', c = nvss_fluxes,cmap = 'coolwarm')

if ras_first is not None:
    plt.scatter(delta_ras_first,delta_decs_first,
    s = 100, label = 'FIRST Sources', c = first_fluxes, cmap = 'coolwarm')

plt.axhline(0)
plt.axvline(0)
plt.xlim(-180,180)
plt.ylim(-180,180)
plt.colorbar()
plt.legend()
plt.xlabel("delta RA")
plt.ylabel("delta Dec")

for i in range(len(ras_nvss)):
    pl.text(delta_ras_nvss[i]+ 5,delta_decs_nvss[i]+ 5, str(i), color = 'k', fontsize = 20)

if ras_first is not None:
    for i in range(len(ras_first)):
        pl.text(delta_ras_first[i] -5,delta_decs_first[i] + 5, str(i), color = 'magenta', fontsize = 20)

plt.show()
