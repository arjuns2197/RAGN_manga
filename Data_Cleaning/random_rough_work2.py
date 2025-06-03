from pdb import line_prefix
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from AGN_FIRST_code import agn_functions as agn

from scipy.special import gammaincc
import astropy
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.table import vstack
import pandas as pd
from scipy import integrate
from scipy import interpolate
import emcee
import time
#log_lambda_max = 2

galaxy_type = 'q'
bin_num = 2


correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]
path = "../AGN_FIRST_files/data/df_M_ssfr_bins/df_" + galaxy_type + str(bin_num) + ".pkl"
df = pd.read_pickle(path)

vd = df['STELLAR_SIGMA_1RE'].to_numpy()
Lr = df['Lr'].to_numpy()
z = df['nsa_redshift'].to_numpy()
sfr = df['log_SFR_SF'].to_numpy()
is_agn = df['agn flux detectable'].to_numpy(dtype = bool)


lambda_lims = agn.get_ER_lim_agn_cav(vd, z, sfr)
lambdas = agn.vector_er_pcav_check_agn(Lr, vd, is_agn)

log_lambda_lims = np.log10(lambda_lims)
log_lambdas = np.log10(lambdas)

inds_lt_minus3 = log_lambda_lims < -3

log_lambda_lims = log_lambda_lims[inds_lt_minus3]

print(np.log10(1/len(log_lambda_lims)))