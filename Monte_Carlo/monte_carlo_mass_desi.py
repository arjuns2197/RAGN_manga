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

log_lambda_max = 1

df_manga = pd.read_pickle("../AGN_FIRST_files/results/manga_radio_agn_matched_to_bgs_y1.pkl")

condition1 = df_manga['IS_SAT'] == False
df_manga = df_manga[condition1]

condition2 = df_manga['log_sfr_sf'] > -100
df_manga = df_manga[condition2]

inputs = np.load('../AGN_FIRST_files/results/MLE/mass_model/mle_mass_final.npz')

alpha_0 = inputs['alpha_0']
beta = inputs['beta']
lambda_star = inputs['lambda_star']
log_lambda_min = inputs['log_lambda_min']

print(alpha_0, beta, lambda_star, log_lambda_min)

vd = df_manga['log_stellar_sigma_1re'].to_numpy()
z = df_manga['nsa_redshift'].to_numpy()
sfr = df_manga['log_sfr_sf'].to_numpy()
mhalo = df_manga['M_HALO'].to_numpy()
nsat = df_manga['N_SAT'].to_numpy()
nsat_q = df_manga['N_SAT_Q'].to_numpy()
nsat_sf = df_manga['N_SAT_SF'].to_numpy()

mass = df_manga['log_mass'].to_numpy()
ssfr = np.log10(10**sfr/10**mass)
plateifu = df_manga['plateifu']

random_ER = []

w = 10

start = time.time()
for i in range(len(sfr)):

    sch = agn.schechter(a= log_lambda_min , b= log_lambda_max ,alpha_0 = alpha_0, beta = beta,
    lambda_star = lambda_star,log_lambda_min= log_lambda_min,prop = mass[i],
    property_string = "log_Mass")

    sample = sch.rvs(size = w)
    random_ER.append(sample)

end = time.time()

print((end-start)/60,"minutes")

temp = []

for i in random_ER:
    for j in i:
        temp.append(j)

random_ER = np.array(temp)
random_ER = 10**random_ER


sfr_large = agn.get_large(sfr, w)
z_large = agn.get_large(z,w)
vd_large = agn.get_large(vd, w)

mhalo_large = agn.get_large(mhalo, w)
nsat_large = agn.get_large(nsat, w)
nsat_q_large = agn.get_large(nsat_q, w)
nsat_sf_large = agn.get_large(nsat_sf, w)

stellar_mass_large = agn.get_large(mass, w)
ssfr_large = agn.get_large(ssfr,w)
Edd_Lumi_large = agn.get_L_Edd(vd_large)

L_sfr_scatter = np.random.normal(scale = 0.28 ,size = len(mass)*w)
Lr_sf = agn.Lsfr_interp(sfr_large) + L_sfr_scatter

pcav_agn = np.log10((random_ER)*Edd_Lumi_large)
Lr_large = np.log10(10**Lr_sf + 10**agn.get_Lr_pcav(pcav_agn))

random_ER_total = agn.get_er_pcav(Lr_large, vd_large)

radio_detectable = agn.vector_flux_detected(sfr_large,Lr_large,z_large,check_agn = False)
agn_exists = agn.vector_is_agn(sfr_large,Lr_large)
agn_detectable = agn.vector_flux_detected(sfr_large,Lr_large,z_large,check_agn = True)

detected_nagn = np.zeros(len(sfr_large), dtype= bool)
detected_nagn[radio_detectable] = np.ones(len(detected_nagn[radio_detectable]), dtype=bool) 
detected_nagn[agn_detectable] = np.zeros(len(detected_nagn[agn_detectable]),dtype = bool)

L_lim = agn.get_L_lim(z_large)
L_lim[radio_detectable] = Lr_large[radio_detectable]

er_lim = agn.get_ER_lim_z(vd_large,z_large)
er_lim[radio_detectable] = random_ER_total[radio_detectable]

plt.scatter(sfr_large[radio_detectable],L_lim[radio_detectable],c='r',s=8)
plt.xlabel("Star-Formation Rate in log(Msun/yr)")
plt.ylabel("Radio Luminosity in log(erg/s)")
plt.ylim(37,42.5)
plt.xlim(-2,2)
plt.legend()
plt.show()

dict = {"log_sfr_sf": sfr_large, "nsa_redshift": z_large, "log_mass" : stellar_mass_large,
"Lr" : Lr_large, "ER" : random_ER_total, "log_ssfr": ssfr_large, "is agn": agn_exists,
"flux detectable": radio_detectable, "agn flux detectable" : agn_detectable,
"detected_nagn":detected_nagn,
"L_lim": L_lim,  "er_lim": er_lim, "log_stellar_sigma_1re": vd_large, "N_SAT" : nsat_large, "M_HALO" : mhalo_large,
"N_SAT_Q": nsat_q_large, "N_SAT_SF": nsat_sf_large}

df_mc_mass = pd.DataFrame(dict)
df_mc_mass.to_pickle("../AGN_FIRST_files/results/MC/mass_model_desi/df_mc_mass_desi_w10.pkl")
