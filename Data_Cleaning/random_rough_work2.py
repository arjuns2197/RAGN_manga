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

#correc_obj = agn.correction_interp()
#np.save("../AGN_FIRST_files/data/interp_obj.npy", correc_obj ,allow_pickle= True)

correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]
df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_new_ERs.pkl")

inds_sf = np.where(df_manga['sSFR'] > -11.5)[0]
inds_q =np.where(df_manga['sSFR'] < -11.5)[0]

plt.scatter(df_manga['log_Mass'][inds_sf], df_manga['log_SFR_SF'][inds_sf], alpha = 0.2, c = 'b')
plt.scatter(df_manga['log_Mass'][inds_q], df_manga['log_SFR_SF'][inds_q], alpha = 0.2, c = 'r')
plt.show()


# df_mc = pd.read_pickle("../AGN_FIRST_files/results/MC/mass_model/df_mc_mass_q_w100.pkl")

# inds_low_mass_agn = np.where(df_mc['agn flux detectable'] & (df_mc['log_Mass'] < 10))[0]

# print(df_mc['log_SFR_SF'][inds_low_mass_agn])

# plt.scatter(df_mc['log_SFR_SF'], df_mc['Lr'], alpha = 0.1)
# plt.scatter(df_mc['log_SFR_SF'][inds_low_mass_agn], df_mc['Lr'][inds_low_mass_agn])
# plt.show()
