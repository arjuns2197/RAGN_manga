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

df_manga = pd.read_pickle("../AGN_FIRST_files/results/manga_radio_agn_matched_to_bgs_y1.pkl")
condition1 = df_manga['IS_SAT'] == False
df_manga = df_manga[condition1]

condition2 = df_manga['log_sfr_sf'] > -100
df_manga = df_manga[condition2]

df_manga.rename(columns={'is_detected_agn': 'agn flux detectable'}, inplace=True)

df_manga['M_HALO'] = np.log10(df_manga['M_HALO'])
df_manga['N_SAT'] = np.log10(df_manga['N_SAT'] + 1e-1)

df_mc = pd.read_pickle("../AGN_FIRST_files/results/MC/mass_model_desi/df_mc_mass_desi_w10.pkl")
df_mc['M_HALO'] = np.log10(df_mc['M_HALO'])
df_mc['N_SAT'] = np.log10(df_mc['N_SAT']+1e-1)

print(df_manga.columns)


def get_agn_frac(property,df_manga,df_mc): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  nbins = 8 #was 8

  sorted_byprop_df = df_mc.sort_values(by = [property])

  binned_byprop_df = pd.cut(sorted_byprop_df[property], bins= nbins)

  sorted_byprop_df['bin_prop'] = binned_byprop_df

  grped_byprop_df = sorted_byprop_df.groupby(["bin_prop"])

  prop_center = []
  f_prop = []

  for name, group in grped_byprop_df:


    string = str(name)
    split = string.split(',')
    left = float(split[0][1:])
    right = float(split[1][:-1])
    center = (left + right)/2
    prop_center.append(center)

    average = group["agn flux detectable"].mean()
    f_prop.append(average)

  f_prop = np.array(f_prop)
  prop_center = np.array(prop_center)

  sorted_byprop_manga = df_manga.sort_values(by = [property])

  binned_byprop_manga = pd.cut(sorted_byprop_manga[property], bins= nbins)

  sorted_byprop_manga['bin prop'] = binned_byprop_manga

  grped_byprop_manga = sorted_byprop_manga.groupby(["bin prop"])

  prop_center_manga = []
  frac_prop_manga = []
  size_prop_manga = []

  for name, group in grped_byprop_manga:
    

    string = str(name)
    split = string.split(',')
    left = float(split[0][1:])
    right = float(split[1][:-1])
    center = (left + right)/2
    prop_center_manga.append(center)
    size_prop_manga.append(len(group))

    average_manga = group["agn flux detectable"].mean()
    frac_prop_manga.append(average_manga)


  size_prop_manga = np.array(size_prop_manga)
  frac_prop_manga = np.array(frac_prop_manga)
  prop_center_manga = np.array(prop_center_manga)

  #print("N galssssssssssssssssssssssssssss:::",size_prop_manga*frac_prop_manga)

  error_low = []
  error_high = []


  for i in range(nbins):

    exp = f_prop[i]*size_prop_manga[i]

    xlo = 0
    xhi = size_prop_manga[i]
    x = np.arange(xlo,xhi,1)


    cdf = agn.betainc(size_prop_manga[i]-x,x+1,1-f_prop[i])
    indices_low = np.where(cdf > 0.16)


    if len(cdf) == 0:
      error_low.append(0.001)


    elif cdf[0] > 0.16:
      error_low.append(f_prop[i])

    elif np.isnan(indices_low).any() == True:
      error_low.append(0.001) 
      
    elif len(indices_low[0]) == 0:
      error_low.append(0.001) 
    
    else:
      error_low.append((exp-indices_low[0][0])/size_prop_manga[i])

    indices_high = np.where(cdf > 0.84)

    if np.isnan(indices_high).any() == True:
      error_high.append(0.001)

    elif len(indices_high) == 0:
      error_high.append(0.001) 

    elif len(indices_high[0]) == 0:
      error_high.append(0.001)

    elif indices_high[0][0] < exp:
      error_high.append(0)

    else:
      error_high.append((indices_high[0][0] - exp)/size_prop_manga[i])

    

  error_low = np.array(error_low)
  error_high = np.array(error_high)
  error_bar_prop = np.array([error_low,error_high])
  
  

  return [frac_prop_manga, prop_center_manga, f_prop, prop_center,error_bar_prop]



def plot_frac(property):

    [frac_prop_manga, prop_manga, frac_prop_mc, prop_mc, errorbar_prop] = get_agn_frac(
        property,df_manga,df_mc)
    
    plt.scatter(prop_manga, frac_prop_manga, label = "Observation", c = 'k', s = 1.5*180)
    plt.errorbar(prop_mc,frac_prop_mc,yerr = errorbar_prop, fmt = 'H',c = 'tab:gray', ms = 1.5*18, capsize = 5, alpha = 0.6, label = "Model")
    plt.legend(prop = {'size' : 15})
    plt.yscale('log')
    plt.ylim(7e-6,1.5)
    plt.yticks([1e-4,1e-3,1e-2,1e-1], labels = [-4,-3,-2,-1])
    plt.tick_params(axis ='y',labelsize = 30)
    plt.ylabel(r"$\log_{\rm 10} ~\rm f_{\rm AGN}$", fontsize = 1.2*30)



plot_frac("N_SAT")
plt.xlabel(r"$\log_{10} \left(N_{\rm SAT} \right)$", fontsize = 30)
plt.tick_params(axis ='x',labelsize = 30)
plt.show()