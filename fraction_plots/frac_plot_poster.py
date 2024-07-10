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
log_lambda_max = 1.7

df_mc = pd.read_pickle("../AGN_FIRST_files/results/MC/mass_model/df_mc_mass_Ha_w1000.pkl")
df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_Ha.pkl")
df_mc_ssfr = pd.read_pickle("../AGN_FIRST_files/results/MC/ssfr_model/df_mc_ssfr_Ha_w1000.pkl")

f = 30

def plot_frac_mass(property):

    [frac_prop_manga, prop_manga, frac_prop_mc, prop_mc, errorbar_prop] = agn.get_agn_frac_Ha(
        property,df_manga,df_mc)
    [frac_prop_manga_1, prop_manga_1, frac_prop_mc_1, prop_mc_1, errorbar_prop_1] = agn.get_agn_frac_Ha(
        property,df_manga,df_mc_ssfr)
    
    plt.figure(figsize = (16,9))
    plt.rcParams['axes.linewidth'] = 6
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.title("RAGN Fraction vs" +  "$~M_\star$", fontsize = 1.5*f)
    plt.scatter(prop_manga, frac_prop_manga, label = "Observation", c = 'k', s = 5*350)
    plt.plot(prop_mc_1,frac_prop_mc_1,c = '#6A6563', linewidth = 4)
    plt.plot(prop_mc,frac_prop_mc,c = 'tab:pink', linewidth = 4)
    plt.errorbar(prop_mc,frac_prop_mc,yerr = errorbar_prop, fmt = 'H',c = 'tab:pink', ms = 5*8, capsize = 20, alpha = 0.45, label = "M Model", linewidth = 6,  capthick = 3)
    plt.errorbar(prop_mc_1,frac_prop_mc_1,yerr = errorbar_prop_1, fmt = 'p',c = '#6A6563', ms = 5*8, capsize = 20, alpha = 0.65, label = "sSFR Model", linewidth = 6, capthick=3)
    plt.legend(prop = {'size' : 30}, loc = 'lower right')
    plt.yscale('log')
    plt.ylim(7e-6,1.5)
    plt.yticks([1e-4,1e-3,1e-2,1e-1], labels = [-4,-3,-2,-1])
    plt.tick_params(axis ='y',labelsize = 2*f, width = 5, length = 10)
    plt.ylabel(r"$\log_{\rm 10} ~\rm f_{\rm AGN}$", fontsize = 2.5*f)
    plt.xlabel(r"$\log_{\rm 10}\left(M_\star/~M_\odot \right)$", fontsize = 2.5*f)
    plt.tick_params(axis ='x',labelsize = 2*f,width = 5, length = 10)
    plt.xlim(8.5,12.5) 
    plt.xticks([9.5,10.5,11.5])
    plt.minorticks_off()
    plt.tight_layout()
    plt.savefig("../Presentations and Talks/radio AGN presentations/STSCI Spring Symposium Poster/Flash Talk/frac_v_mass.jpg", dpi = 300)
    plt.show()




def plot_frac_ssfr(property):

    [frac_prop_manga, prop_manga, frac_prop_mc, prop_mc, errorbar_prop] = agn.get_agn_frac_Ha(
        property,df_manga,df_mc)
    [frac_prop_manga_1, prop_manga_1, frac_prop_mc_1, prop_mc_1, errorbar_prop_1] = agn.get_agn_frac_Ha(
        property,df_manga,df_mc_ssfr)
    
    plt.figure(figsize = (16,9))
    plt.rcParams['axes.linewidth'] = 6
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.title("RAGN Fraction vs sSFR", fontsize = 1.5*f)

    plt.scatter(prop_manga, frac_prop_manga, label = "Observation", c = 'k', s = 5*350)
    plt.plot(prop_mc_1,frac_prop_mc_1,c = '#6A6563', linewidth = 4)
    plt.plot(prop_mc,frac_prop_mc,c = 'tab:pink', linewidth = 4)
    plt.errorbar(prop_mc,frac_prop_mc,yerr = errorbar_prop, fmt = 'H',c = 'tab:pink', ms = 5*8, capsize = 20, alpha = 0.45, label = "M Model", linewidth = 6,  capthick = 3)
    plt.errorbar(prop_mc_1,frac_prop_mc_1,yerr = errorbar_prop_1, fmt = 'p',c = '#6A6563', ms = 5*8, capsize = 20, alpha = 0.65, label = "sSFR Model", linewidth = 6, capthick=3)
    plt.legend(prop = {'size' : 30}, loc = 'lower left')
    plt.yscale('log')
    plt.ylim(7e-6,1.5)
    plt.yticks([1e-4,1e-3,1e-2,1e-1], labels = [-4,-3,-2,-1])
    plt.tick_params(axis ='y',labelsize = 2*f, width = 5, length = 10)
    plt.ylabel(r"$\log_{\rm 10} ~\rm f_{\rm AGN}$", fontsize = 2.5*f)
    plt.xlabel(r"$\log_{\rm 10}\left(\rm sSFR/ \rm ~yr^{-1} \right)$", fontsize = 2.5*f)
    plt.tick_params(axis ='x',labelsize = 2*f,width = 5, length = 10)
    plt.xlim(-16,-9.2)    
    plt.xticks([-15,-13.5,-12,-10.5])
    plt.savefig("../Presentations and Talks/radio AGN presentations/STSCI Spring Symposium Poster/Flash Talk/frac_v_ssfr.jpg", dpi = 300)
    plt.minorticks_off()
    plt.tight_layout()
    plt.show()


plot_frac_mass('log_Mass')
plot_frac_ssfr('sSFR')