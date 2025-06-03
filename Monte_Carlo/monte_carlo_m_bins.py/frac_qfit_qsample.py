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

df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_new_ERs.pkl")
df_mc1 = pd.read_pickle("../AGN_FIRST_files/results/MC/mass_bins/df_mc_qfit_qsample" +str(1) + "_w10.pkl")
df_mc2 = pd.read_pickle("../AGN_FIRST_files/results/MC/mass_bins/df_mc_qfit_qsample" +str(2) + "_w10.pkl")
df_mc3 = pd.read_pickle("../AGN_FIRST_files/results/MC/mass_bins/df_mc_qfit_qsample" +str(3) + "_w10.pkl")
df_mc4 = pd.read_pickle("../AGN_FIRST_files/results/MC/mass_bins/df_mc_qfit_qsample" +str(4) + "_w10.pkl")


dfs_mc = [df_mc1, df_mc2, df_mc3, df_mc4]

def plot_frac(property):

    [frac_prop_manga, prop_manga, frac_prop_mc, prop_mc, errorbar_prop] = agn.get_agn_frac_mbins(
    property,df_manga,dfs_mc,4,'q')
    
    plt.scatter(prop_manga, frac_prop_manga, label = "Observation", c = 'k', s = 1.5*180)
    plt.errorbar(prop_mc,frac_prop_mc,yerr = errorbar_prop, fmt = 'H',c = 'tab:gray', ms = 1.5*18, capsize = 5, alpha = 0.6, label = "Model")
    plt.legend(loc = 'lower left',prop = {'size' : 15})
    plt.yscale('log')
    plt.ylim(7e-6,1.5)
    plt.yticks([1e-4,1e-3,1e-2,1e-1], labels = [-4,-3,-2,-1])
    plt.tick_params(axis ='y',labelsize =f)
    plt.ylabel(r"$\log_{\rm 10} ~\rm f_{\rm AGN}$", fontsize = 1.2*f)


f = 30
plt.figure(figsize = (7.2,8))
plt.suptitle("Q fits Q sample", fontsize = f)

properties = ["log_Mass","STELLAR_SIGMA_1RE","log_SFR_SF","sSFR"]

plt.subplots_adjust(hspace = 0.5, wspace = 0.25)

for i in enumerate(properties):
    plt.subplot(2,2,i[0]+1)

    if i[0] == 0:
        
        plt.xlabel(r"$\log_{\rm 10}\left(\rm M_\star/ \rm ~M_\odot \right)$", fontsize = f)
        plt.tick_params(axis ='x',labelsize =f)
        plt.xlim(9,12.5) 
        plt.xticks([9.5,10.5,11.5])
    
    if i[0] == 1:
        
        plt.xlabel(r"$\log_{\rm 10}\left(\sigma_v/\rm ~km~ \rm s^{\rm -1}\right)$", fontsize = f)
        plt.tick_params(axis ='x',labelsize =f)
        plt.xlim(1.7,2.7)
        plt.xticks([1.8,2,2.2,2.4,2.6]) 
        
    if i[0] == 2:
        
        plt.xlabel(r"$\log_{10}\left(\rm SFR/\rm ~M_\odot ~ \rm yr^{\rm -1}\right)$", fontsize = f)
        plt.tick_params(axis ='x',labelsize =f)
        plt.xlim(-6,2.5)
        plt.xticks([-5,-3,-1,1])


    if i[0] == 3:
        
        plt.xlabel(r"$\log_{\rm 10}\left(\rm sSFR/ ~ \rm yr^{\rm -1}\right)$", fontsize = f)
        plt.tick_params(axis ='x',labelsize =f)
        plt.xlim(-16,-9.2)    
        plt.xticks([-15,-13.5,-12,-10.5])


    plot_frac(i[1])


plt.show()
