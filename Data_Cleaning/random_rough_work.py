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




df_matched = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6731.pkl")
correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]


df_Ha = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_Ha.pkl")


'''
path_low_optical = '../AGN_FIRST_files/data/paper_pix/low_mass_optical.png'
low_optical = iio.imread('../AGN_FIRST_files/data/paper_pix/low_mass_optical.png')
path_low_nvss = '../AGN_FIRST_files/data/paper_pix/low_mass_nvss.fits'
low_nvss = -fits.open(path_low_nvss)[0].data
path_low_first = '../AGN_FIRST_files/data/paper_pix/low_mass_first.fits'
low_first = -fits.open(path_low_first)[0].data

path_mid_optical = '../AGN_FIRST_files/data/paper_pix/mid_mass_optical.png'
mid_optical = iio.imread('../AGN_FIRST_files/data/paper_pix/mid_mass_optical.png')
path_mid_nvss = '../AGN_FIRST_files/data/paper_pix/mid_mass_nvss.fits'
mid_nvss = -fits.open(path_mid_nvss)[0].data
path_mid_first = '../AGN_FIRST_files/data/paper_pix/mid_mass_first.fits'
mid_first = -fits.open(path_mid_first)[0].data

path_high_optical = '../AGN_FIRST_files/data/paper_pix/low_high_optical.png'
high_optical = iio.imread('../AGN_FIRST_files/data/paper_pix/high_mass_optical.png')
path_high_nvss = '../AGN_FIRST_files/data/paper_pix/high_mass_nvss.fits'
high_nvss = -fits.open(path_high_nvss)[0].data
path_high_first = '../AGN_FIRST_files/data/paper_pix/high_mass_first.fits'
high_first = -fits.open(path_high_first)[0].data


images = [low_optical, low_nvss[0][0], low_first,
mid_optical, mid_nvss[0][0], mid_first,
high_optical, high_nvss[0][0], high_first]
 
plt.figure()


for i in range(9):
    plt.subplot(3,3,i+1)
    plt.subplots_adjust(hspace = 0, wspace = 0)
    plt.imshow(images[i], cmap='gray', aspect = 'auto')
    plt.plot([1,2,3,4,5,7,8,9], [1,2,3,4,5,7,8,9], linewidth = 4)
    plt.xticks([])
    plt.yticks([])

plt.show()


sfr = df_matched['log_SFR_ssp']
M =  df_matched['log_Mass']

inds = df_matched['agn flux detectable']
'''


df_vac = pd.read_pickle("../AGN_FIRST_files/results/VAC/df_vac.pkl")
df = fits.open('../AGN_FIRST_files/results/VAC/VAC.fits')[1].data
inds_erd = df['in_erd_analysis']

df_vac = df_vac[inds_erd]
vds = df_vac['STELLAR_SIGMA_1RE']

inds_agn_vac = df_vac['is_agn'].to_numpy(dtype = bool)
correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]


plt.scatter(df_vac['log_SFR_ssp'][inds_agn_vac], df_vac['Lr'][inds_agn_vac])
plt.show()

mass_model_para = agn.mle_prop("log_Mass",df_vac,correc_obj,False)
alpha_0, beta, lambda_star, log_lambda_min = mass_model_para

print(alpha_0,beta, lambda_star,log_lambda_min)
