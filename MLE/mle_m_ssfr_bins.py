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

correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]

df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_new_ERs.pkl")

inds_mass_cut = np.where(df_manga['log_Mass'] > 10)
df_manga = df_manga.iloc[inds_mass_cut]

nbins = 5

# inds_sf = np.where(df_manga['sSFR'] > -11.5)[0]
# inds_q = np.where(df_manga['sSFR'] < -11.5)[0]

# df_qs = df_manga.iloc[inds_q]
# agn.group_by_mass(df_qs,nbins,'q')
# agn.mle_bins('q', nbins, correc_obj)

# df_sfs = df_manga.iloc[inds_sf]
# agn.group_by_mass(df_sfs,nbins,'sf')
# agn.mle_bins('sf', nbins, correc_obj)

path_ = '../AGN_FIRST_files/results/MLE/mass_ssfr_bins/mle_q'

for i in range(nbins):

    bin_num = i + 1 
    path = path_ + str(bin_num) + ".npz"

    model_parms = np.load(path)
    M = model_parms['M']
    alpha_0 = model_parms['alpha_0'] + 1e-7
    lambda_star = model_parms['lambda_star']
    log_lambda_min = model_parms['log_lambda_min']

    optimal_Fagn = agn.prob_agnness_unif(alpha_0,10**lambda_star,log_lambda_min,-3,correc_obj)

    print(alpha_0, lambda_star, log_lambda_min, np.log10(optimal_Fagn))
