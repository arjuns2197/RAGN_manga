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


mass_model_para = agn.mle_prop_new_ers("STELLAR_SIGMA_1RE",df_manga,correc_obj,False)
alpha_0, beta, lambda_star, log_lambda_min = mass_model_para

print(alpha_0,beta, lambda_star,log_lambda_min)

### Save Results
np.savez_compressed('../AGN_FIRST_files/results/MLE/vd_model/mle_vd_final.npz',alpha_0 = alpha_0, beta = beta, lambda_star = lambda_star,log_lambda_min = log_lambda_min)