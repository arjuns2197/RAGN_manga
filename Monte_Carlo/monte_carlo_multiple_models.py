from random import random
from scipy.stats import rv_continuous

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
import time

df_manga = pd.read_pickle('../AGN_FIRST_files/data/df_matched_6061.pkl')


vd = df_manga['vel_disp_ssp_cen'].to_numpy()
z = df_manga['nsa_redshift'].to_numpy()
sfr = df_manga['log_SFR_ssp'].to_numpy()

lambda_lim_all = agn.get_ER_lim_z(vd,z,sfr)
mass = df_manga['log_Mass'].to_numpy()
es_weights = df_manga['ESWEIGHT'].to_numpy()
ssfr = np.log10(10**sfr/10**mass)


inputs = np.load('../AGN_FIRST_files/results/MCMC/random_central_samples.npz')

alpha_0s = inputs['alpha0s']
betas = inputs['betas']
lambda_stars = inputs['lamstars']
log_lambda_mins = inputs['log_mins']


for j in range(len(alpha_0s)):

    print(alpha_0s[j],betas[j],lambda_stars[j],log_lambda_mins[j])
    random_ER = []

    w = 10

    start = time.time()
    for i in range(len(sfr)):

        sch = agn.schechter(a= log_lambda_mins[j] , b= 1.7 ,alpha_0 = alpha_0s[j], beta = betas[j],
        lambda_star = lambda_stars[j],log_lambda_min= log_lambda_mins[j],prop = mass[i],
        property_string = "log_Mass")

        sample = sch.rvs(size = w)
        random_ER.append(sample)

    end = time.time()

    print((end-start)/60,"minutes")

    temp = []

    for k in random_ER:
        for l in k:
            temp.append(l)

    random_ER = np.array(temp)

    vd_large = agn.get_large(vd, w)
    sfr_large = agn.get_large(sfr, w)
    z_large = agn.get_large(z,w)
    ER_lim = agn.get_large(lambda_lim_all,w)
    stellar_mass_large = agn.get_large(mass, w)
    ssfr_large = agn.get_large(ssfr,w)
    es_weights_large = agn.get_large(es_weights,w)
    Edd_Lumi_all = agn.get_L_Edd(vd_large)


    error = np.random.normal(scale = 0.28 ,size = len(lambda_lim_all)*w)
    Lr_sf = agn.Lsfr_interp(sfr_large) + error
    sf_ER = agn.get_ER(vd_large,Lr_sf)

    bolo = (10**random_ER+sf_ER)*Edd_Lumi_all
    Lx = np.log10(bolo/20)
    Lr_large = (Lx-7.1)/0.925

    radio_detectable = agn.vector_flux_detected(sfr_large,Lr_large,z_large, check_agn = False)
    agn_exists = agn.vector_is_agn(sfr_large,Lr_large)
    agn_detectable = agn.vector_flux_detected(sfr_large,Lr_large,z_large,check_agn = True)
    ind_dets = agn.vector_flux_detected(sfr_large,Lr_large,z_large,False)
    high_er_inds = agn.vector_high_er(random_ER)

    plt.scatter(sfr_large[ind_dets],Lr_large[ind_dets],c='r',s=8)
    plt.xlabel("Star-Formation Rate in log(Msun/yr)")
    plt.ylabel("Radio Luminosity in log(erg/s)")
    plt.ylim(37,42.5)
    plt.xlim(-2,2)
    plt.legend()
    plt.show()


    dict = {"log_SFR_ssp": sfr_large, "nsa_redshift": z_large, "log_Mass" : stellar_mass_large,
    "Lr" : Lr_large, "Random ER" : random_ER, "sSFR": ssfr_large, "is agn": agn_exists,
    "flux detectable": radio_detectable, "agn flux detectable" : agn_detectable,
    "vel_disp_ssp_cen" : vd_large, "ESWEIGHT" : es_weights_large,"is High ER": high_er_inds}

    df_mc_mass = pd.DataFrame(dict)

    df_mc_mass.to_pickle(f'../AGN_FIRST_files/results/MC/multiple_models/df_mc_{j}th_model.pkl')

  