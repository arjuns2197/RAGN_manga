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

#df_manga = agn.load_matched_df_new_ERs()
#df_manga.to_pickle("../AGN_FIRST_files/data/df_matched_6255_new_ERs.pkl")
df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_new_ERs.pkl")

inds_sf = np.where(df_manga['sSFR'] > -11.5)[0]
df_manga_sf = df_manga.iloc[inds_sf]

mass_model_para = agn.mle_prop_new_ers("log_Mass",df_manga_sf,correc_obj,False)
alpha_0_sf, beta_sf, lambda_star_sf, log_lambda_min_sf = mass_model_para



inds_q = np.where(df_manga['sSFR'] < -11.5)[0]
df_manga_q = df_manga.iloc[inds_q]

mass_model_para = agn.mle_prop_new_ers("log_Mass",df_manga_q,correc_obj,False)
alpha_0_q, beta_q, lambda_star_q, log_lambda_min_q = mass_model_para

print(alpha_0_sf,beta_sf, lambda_star_sf,log_lambda_min_sf)
print(alpha_0_q,beta_q, lambda_star_q,log_lambda_min_q)

### Save Results
np.savez_compressed('../AGN_FIRST_files/results/MLE/mass_model/mle_mass_sfq_split.npz',
alpha_0_sf = alpha_0_sf, beta_sf = beta_sf, lambda_star_sf = lambda_star_sf,log_lambda_min_sf = log_lambda_min_sf,
alpha_0_q = alpha_0_q, beta_q = beta_q, lambda_star_q = lambda_star_q,log_lambda_min_q = log_lambda_min_q)