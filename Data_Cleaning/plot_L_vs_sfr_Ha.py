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


#df_manga = agn.load_matched_df()
#df_manga.to_pickle("../AGN_FIRST_files/data/df_matched_6061.pkl")

df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_Ha.pkl")

Lr_manga = df_manga["Lr"] 
sfr_manga = df_manga["log_SFR_SF"]
lam_man = df_manga["ER"]
ind_agn_man = df_manga["is agn"].to_numpy(dtype= bool)
ssfr_manga = df_manga["sSFR"]

num_agn = len(np.where(ind_agn_man == True)[0])
print("Number of AGN: ", num_agn)

inds_mats = np.where(df_manga['Flux'] > 1)[0]
print("Number of matches: ",len(inds_mats))

f = 30
sfr_line = np.linspace(-5,1.75,200)

sfr_left = np.linspace(-5,0.044,200)
sfr_right = np.linspace(0.044,1.75,200)

agn_lcut = 38.6*np.ones(200)
agn_rcut = agn.Lsfr_interp(sfr_right) + 1

## Observed Lr vs SFR

plt.figure(figsize =(20,10))
plt.scatter(sfr_manga,Lr_manga, c = 'k')
#plt.plot(sfr_line, agn.Lsfr_interp_hopkins(sfr_line), c='k', label = "Hopkins 2003", linewidth = 4, linestyle = "--")
plt.plot(sfr_line, agn.Lsfr_interp(sfr_line), c='k', label = "Hopkins with Chabrier Offset", linewidth = 4)
plt.plot(sfr_left, agn_lcut, c='tab:gray', label = "AGN cut", linewidth = 10, alpha = 0.3)
plt.plot(sfr_right, agn_rcut, c='tab:gray', linewidth = 10, alpha = 0.3)
plt.xticks([-4.5,-3,-1.5,0,1.5],fontsize = f)
plt.yticks([38,39,40,41], fontsize = f)
plt.ylabel(r"$\log_{\rm 10}\left(\rm L_{\rm 1.4 ~GHz}/\rm erg~s^{\rm -1}\right)$", fontsize = f)
plt.xlabel(r"$\log_{\rm 10}\left(\rm SFR/\rm M_{\odot}~yr^{\rm -1}\right)$", fontsize = f)
plt.ylim(37,42)
plt.xlim(-5,2)
plt.legend(prop = {'size':20})
plt.savefig("../AGN_FIRST_files/results/plots/L_vs_SFR/L_vs_SFR_Ha(agn_cut).pdf")
plt.show()