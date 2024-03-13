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
from astropy.table import Table

df_matched = pd.read_pickle("../AGN_FIRST_files/data/df_drpmatched.pkl")

[seps_near, seps_far, ind_sources, ind_vi, ind_cands] = agn.get_nvss_2_seps(df_matched)
seps = np.linspace(0,90,2000)
cut = agn.v_nvss_double_candidate_cut(seps)

xticks = [0,30,60,90]
yticks = [30,60,90]
f = 35

plt.scatter(seps_near,seps_far,c = 'tab:gray',s =120)
plt.scatter(seps_near[ind_cands],seps_far[ind_cands],c = 'k',s =100, label = 'Candidate NVSS Double')
plt.scatter(seps_near[ind_sources],seps_far[ind_sources], facecolors = 'none', edgecolors = 'k', marker = 's', s = 250, label = "Accepted NVSS Double")
plt.plot(seps,cut, c = 'k', label = "Candidate Cut")
plt.scatter(seps_near[ind_vi],seps_far[ind_vi], facecolors = 'none', edgecolors = 'k', marker = 'D',
s = 250, label = "Needs VI")
plt.plot(seps,seps, c = 'k')
plt.ylim(0,120)
plt.xlim(0,120)
plt.xticks(xticks,fontsize = f)
plt.yticks(yticks, fontsize = f)
plt.xlabel(r"$\rm Nearer~ Source~ Offset/ ~arcsec$", fontsize = f)
plt.ylabel(r"$\rm Farther~ Source~ Offset/~ arcsec$", fontsize = f)
plt.legend()
plt.show()





