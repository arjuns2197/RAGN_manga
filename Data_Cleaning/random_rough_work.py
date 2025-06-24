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
from scipy.stats import rv_continuous
from scipy.stats import norm
import time
from scipy.special import gammaincc
from scipy.special import gammainc
from scipy.special import gamma
from scipy.stats import poisson
from astropy import cosmology
from scipy.optimize import curve_fit
import imageio as iio
import requests
from PIL import Image
import matplotlib.patches as patches
f = 45

galaxy_type = 'sf'
bin_num = 1

group = pd.read_pickle("../AGN_FIRST_files/data/df_M_ssfr_bins/df_" + galaxy_type + str(bin_num) + ".pkl")

mass = group['log_Mass']
if type(mass) != np.ndarray:
    mass = mass.to_numpy()

Lr = group['Lr']
if type(Lr) != np.ndarray:
    Lr = Lr.to_numpy()

sfr = group['log_SFR_SF']
if type(sfr) != np.ndarray:
    sfr = sfr.to_numpy()

vd = group['STELLAR_SIGMA_1RE']
if type(vd) != np.ndarray:
    vd = vd.to_numpy()

z = group['nsa_redshift']
if type(z) != np.ndarray:
    z = z.to_numpy()


lambda_lim = agn.get_ER_lim_agn_cav(vd,z,sfr)
condition = lambda_lim < 10**(-3)
lambda_lim = lambda_lim[condition]

Ngal = len(lambda_lim)

print(agn.get_Fagn_upperlim(Ngal))