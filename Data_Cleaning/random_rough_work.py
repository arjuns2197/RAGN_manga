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






correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]
df_Ha = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_Ha.pkl")

inputs = np.load('../AGN_FIRST_files/results/MLE/mass_model/mle_mass_Ha_6255.npz')

alpha_0 = inputs['alpha_0']
beta = inputs['beta']
lambda_star = inputs['lambda_star']
log_lambda_min = inputs['log_lambda_min']
log_lambda_c = -2


for i in range(6255):

    if df_Ha['is agn'][i] == True:
        if df_Ha['log_Mass'][i] < 10.45:
            print(df_Ha['plateifu'][i])
            print(np.log10(df_Ha['ER'][i]))




a = [1]
b = [1]

plt.plot(a,b, 'H')
plt.show()