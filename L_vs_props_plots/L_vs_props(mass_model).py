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
from scipy.stats import poisson
log_lambda_max = 1



# Load Results

df_mc = pd.read_pickle("../AGN_FIRST_files/results/MC/mass_model/df_mc_mass_w1.pkl")
df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6731.pkl")


sfr_manga = df_manga['log_SFR_ssp']

w = 1 # ratio of montcarlo sample size to manga sample size
sfr = df_mc['log_SFR_ssp']
z = df_mc['nsa_redshift']
Lr = df_mc["Lr"]
lam = df_mc["ER"]
ssfr = df_mc['sSFR']
mass = df_mc['log_Mass']
vd = df_mc['STELLAR_SIGMA_1RE']
Lr_manga = df_manga["Lr"] 

f = 30

# MOnte Carlo Lradio vs Property plots

ind_flux_det_mc = df_mc['flux detectable'].to_numpy(dtype= bool)
ind_flux_det_manga = df_manga['flux detectable'].to_numpy(dtype= bool)

ind_det_agn_mc = df_mc['agn flux detectable'].to_numpy(dtype= bool)
ind_det_agn_manga = df_manga['agn flux detectable'].to_numpy(dtype= bool)

properties = [mass,vd,sfr,ssfr,mass,vd,sfr,ssfr]
x_lims = [(9,12.2),(1.7,2.7),(-2,1.9),(-13.2,-9.5)]
axes_labels = [r"$\log_{\rm 10}(\rm M_{\star}/\rm M_{\odot})$", r"$\log_{\rm 10}(\sigma_{v}/\rm ~km~ \rms^{\rm -1})$",
r"$\log_{\rm 10}(\rm SFR/\rm M_{\odot}~yr^{\rm -1})$", r"$\log_{\rm 10}(\rm sSFR/~\rm yr^{\rm -1})$"]

f = 30 #fontsize
yticks = [38,39,40,41]
xticks = [[9.5,10.5,11.5],[1.8,2.2,2.6],[-1,0,1],[-12.5,-10.5]]
plt.figure(figsize= (7.2,8))
plt.subplots_adjust(wspace=0,hspace =0)
plt.suptitle(r"$\rm M_\star$ Model", fontsize = f)


for i in enumerate(properties):
  if i[0] < 4:
    plt.subplot(2,4,i[0]+1)
    plt.scatter(i[1][ind_flux_det_manga],Lr_manga[ind_flux_det_manga], c= 'tab:gray', label = "Observation")
    plt.scatter(i[1][ind_det_agn_manga],Lr_manga[ind_det_agn_manga], c= 'k', label = "Detected AGN")

    if i[0] == 0:
      plt.ylabel(r"$\log_{\rm 10}(\rm L_{\rm 1.4 ~GHz}/~\rm erg~\rm s^{\rm -1})$", fontsize = 0.75*f)
      plt.yticks(yticks,fontsize = f)
      plt.legend(prop={'size': 15})

    elif i[0] == 2:
      sfr_line = np.linspace(-2,1.75,200)

      sfr_left = np.linspace(-2,0.044,200)
      sfr_right = np.linspace(0.044,1.75,200)

      agn_lcut = 38.6*np.ones(200)
      agn_rcut = agn.Lsfr_interp(sfr_right) + 1
      plt.plot(sfr_left, agn_lcut, c='k', label = "AGN cut", linewidth = 5)
      plt.plot(sfr_right, agn_rcut, c='k', linewidth = 5)
      plt.yticks([])
      
    else:
      plt.yticks([])
    plt.tick_params(axis = 'y',which = 'both', bottom = False, top = False, labelbottom = False)
    plt.xlim(x_lims[i[0]])
    plt.ylim(37,42)
    plt.xticks([])

  else:
    plt.subplot(2,4,i[0]+1)
    plt.scatter(i[1][ind_flux_det_mc], Lr[ind_flux_det_mc], c = 'tab:gray', label = "Model")
    plt.xlabel(axes_labels[i[0]-4], fontsize = 0.9*f)
    plt.scatter(i[1][ind_det_agn_mc],Lr[ind_det_agn_mc], c= 'k', label = "Detected AGN")

    if i[0] -4 == 0:
      plt.ylabel(r"$\log_{\rm 10}(\rm L_{\rm 1.4~ GHz}/~\rm erg~\rm s^{\rm -1})$", fontsize = 0.75*f)
      plt.yticks(yticks,fontsize = f)
      plt.legend(prop={'size': 15})
      

    elif i[0] -4 == 2:
      sfr_line = np.linspace(-2,1.75,200)

      sfr_left = np.linspace(-2,0.044,200)
      sfr_right = np.linspace(0.044,1.75,200)

      agn_lcut = 38.6*np.ones(200)
      agn_rcut = agn.Lsfr_interp(sfr_right) + 1
      plt.plot(sfr_left, agn_lcut, c='k', label = "AGN cut", linewidth = 5)
      plt.plot(sfr_right, agn_rcut, c='k', linewidth = 5)    
      plt.yticks([])

    else:
      plt.yticks([])
    plt.xlim(x_lims[i[0]-4])
    plt.ylim(37,42)
    plt.xticks(xticks[i[0]-4],fontsize = f)

plt.show()