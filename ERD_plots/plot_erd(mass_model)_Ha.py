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

# Load dataframes

df_mc = pd.read_pickle("../AGN_FIRST_files/results/MC/mass_model/df_mc_mass_w1000_Ha.pkl")
w = 1000 #mc_results['w'] # ratio of montcarlo sample size to manga sample size
df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_Ha.pkl")

# load galaxy properties

## Monte Carlo
sfr_mc = df_mc['log_SFR_SF']
z_mc = df_mc['nsa_redshift']
Lr_mc = df_mc["Lr"]
er_mc = df_mc["ER"]
ssfr_mc = df_mc['sSFR']
mass_mc = df_mc['log_Mass']


ind_agn_det_mc = df_mc['agn flux detectable'].to_numpy(dtype=bool)

plt.scatter(mass_mc[ind_agn_det_mc],Lr_mc[ind_agn_det_mc])
plt.show()
## Real

sfr_manga = df_manga['log_SFR_SF']
z_manga = df_manga['nsa_redshift']
Lr_manga = df_manga["Lr"]
er_manga = df_manga["ER"]
ssfr_manga = df_manga['sSFR']
mass_manga = df_manga['log_Mass']



ind_agn_det_manga = df_manga['agn flux detectable'].to_numpy(dtype=bool)

ind_dets_mc = df_mc["agn flux detectable"].to_numpy(dtype= bool)



bin_min = -4 #(This is the lowest value of detected ER across the MC samples) #df_mc["Random ER"][ind_dets_mc].min()
bin_max = np.log10(df_manga["ER"][ind_agn_det_manga]).max()

bin_locs = np.linspace(bin_min, bin_max, 6)

print("Hiiiiiiiiiiiiiiiiiii",np.log10(df_manga["ER"][ind_agn_det_manga].min()))
print("Hiiiiiiiiiiiiiiiiiii",np.log10(df_mc["ER"][ind_agn_det_mc].min()))

# Sort properties into bins

def bin_property(property, nbins, df):

  sorted_df_prop = df.sort_values(by = [property])
  binned_df_prop = pd.cut(sorted_df_prop[property], bins= nbins)
  sorted_df_prop['bin_prop'] = binned_df_prop
  grped_df_prop = sorted_df_prop.groupby(["bin_prop"], as_index = True)

  return grped_df_prop


def erd_bins(property, nbins):

    '''Returns the real and MC ERDs in
    bins of given property.
    
    First 'nbins' returned elements correspond to real ERDs and the last to MC.

    Inputs a string of the exact property name in the df and the required
    number of bins '''
    
    prop_bins_manga = bin_property(property,nbins,df_manga)
    prop_bins_mc = bin_property(property,nbins,df_mc)

    hists_prop = [] # First 'nbins' entries will correspond to manga and the last to MC


    for name, group in prop_bins_manga:

    
      ind_agn = group["agn flux detectable"].to_numpy(bool)
      detected_agn = np.log10(group["ER"][ind_agn].to_numpy())

      hists_prop.append(detected_agn)



    for name, group in prop_bins_mc:

      ind_agn = group["agn flux detectable"].to_numpy(bool)
      detected_agn = np.log10(group["ER"][ind_agn].to_numpy())

      hists_prop.append(detected_agn)

    
    return hists_prop


f = 30 ##Fontsize

def pois_errbrs(mu):

  ''' Returns the 1 sigma lower and upper bounds of the poisson distribution for 
  the given mu. Uses the "pois_cdf" 2D array generated earlier.'''

  x_low = poisson.ppf(0.16,mu)
  x_high = poisson.ppf(0.84,mu)

  if x_low == 0:

    ebar_low = np.log10(mu) - np.log10(0.001)
  
  if x_low != 0:

    ebar_low = np.log10(mu) - np.log10(x_low)


  if x_high == 0:

    ebar_high = np.log10(x_high) - np.log10(0.001)
    
  if x_high != 0:
  
    ebar_high = np.log10(x_high) - np.log10(mu)

  return np.array([ebar_low, ebar_high])

def chi_sq(real,mc):

  chi2 = []

  for i in range(len(mc)):

    if mc[i] > 0:

      chi2.append((real[i]-mc[i])**2/mc[i])
  
  chi2 = np.array(chi2)

  return chi2.sum()


def plot_erd_bin(property): 

  if property == "log_SFR_SF":
    prop_label = "SFR"

  elif property == "sSFR":
    prop_label = "sSFR"

  elif property == "log_Mass":
    prop_label = r"$\rm M_{\star}$"

  elif property == "STELLAR_SIGMA_1RE":
    prop_label = r"$\rm \sigma_{\rm v}$"

  erds = erd_bins(property,2)
  erd_prop_real = erds[0:2] 
  erd_prop_mc = erds[2:]


  low_real, bins_low_real = np.histogram(erd_prop_real[0], bins = bin_locs)
  bins_low_real = 0.5*(bins_low_real[:-1] + bins_low_real[1:])
  high_real, bins_high_real = np.histogram(erd_prop_real[1], bins = bin_locs)
  bins_high_real = 0.5*(bins_high_real[:-1] + bins_high_real[1:])


  low_mc, bins_low_mc = np.histogram(erd_prop_mc[0], bins = bin_locs, weights= np.ones(len(erd_prop_mc[0]))/w)
  bins_low_mc = 0.5*(bins_low_mc[:-1] + bins_low_mc[1:])
  high_mc, bins_high_mc = np.histogram(erd_prop_mc[1], bins = bin_locs, weights= np.ones(len(erd_prop_mc[1]))/w)
  bins_high_mc = 0.5*(bins_high_mc[:-1] + bins_high_mc[1:])


  '''print("Beginning of ", prop_label)
  chi2_low = chi_sq(low_real,low_mc)
  print("Chi square low Q",chi2_low)
  chi2_high = chi_sq(high_real,high_mc)
  print("Chi square high Q",chi2_high)
  print("End of ", prop_label)'''

  low_bin_ebars = []
  high_bin_ebars = []

  for i in range(len(low_mc)):

    if low_mc[i] < 0.3: #0.3

      low_mc[i] = 0

    low_bin_ebars.append(pois_errbrs(low_mc[i]))

  low_bin_ebars = np.array(low_bin_ebars)

  
  for j in range(len(high_mc)):

    high_bin_ebars.append(pois_errbrs(high_mc[j]))
      
  high_bin_ebars = np.array(high_bin_ebars)

  low_bin_ebars_lows = low_bin_ebars[:,0]
  low_bin_ebars_highs = low_bin_ebars[:,1]

  high_bin_ebars_lows =  high_bin_ebars[:,0] 
  high_bin_ebars_highs = high_bin_ebars[:,1]

  low_bin_ebars = [low_bin_ebars_lows, low_bin_ebars_highs]
  high_bin_ebars = [high_bin_ebars_lows, high_bin_ebars_highs]
  
  print("low real",low_real)

  plt.plot(bins_low_real,np.log10(low_real), label = "Observation" + " (low "+ prop_label+")", linewidth = 1.5*4, c = 'k')
  plt.plot(bins_low_mc,np.log10(low_mc), label = "Model" + " (low "+prop_label+")", linestyle = "--",linewidth = 1.5*4, c= 'k')
  plt.errorbar(x = bins_low_mc, y = np.log10(low_mc), yerr = low_bin_ebars, fmt = 'H', c = "k", linewidth = 3, capsize = 5)
  plt.xlim(-4,2)
  plt.ylim(-0.5,3.5)
  plt.plot(bins_high_real,np.log10(high_real), label = "Observation" + " (high "+prop_label+")",linewidth = 1.5*4, c= 'tab:gray')
  plt.plot(bins_high_mc,np.log10(high_mc) ,label = "Model" + " (high "+prop_label+")" ,linestyle = "--",linewidth = 1.5*4, c= 'tab:gray')
  plt.errorbar(x = bins_high_mc, y = np.log10(high_mc), yerr = high_bin_ebars, fmt = 'H', c= "tab:gray", linewidth = 3, capsize = 5)
  plt.legend(prop = {'size' : 15})

  if property == "log_Mass":
    annotation = r"$M_{\star}$"
  
  elif property == "STELLAR_SIGMA_1RE":
    annotation = r"$\sigma_{\rm v}$"

  elif property == "log_SFR_SF":
    annotation = r"$\rm SFR_{H\alpha}$"
  
  else:
    annotation = r"$\rm sSFR_{H\alpha}$"

  plt.annotate(annotation,(-3,2.5), fontsize = 1.5*30)
  

plt.figure(figsize = (10,10))
plt.suptitle(r"$\rm M_\star$ Model", fontsize = f)

properties = ["log_Mass","STELLAR_SIGMA_1RE","log_SFR_SF", "sSFR"]

plt.subplots_adjust(hspace = 0, wspace = 0)

for i in enumerate(properties):
  plt.subplot(2,2,i[0]+1)

  if i[0] == 0 or i[0] == 2:
    plt.ylabel(r"$\log_{\rm 10}(\rm N_{\rm AGN})$", fontsize = f)
    plt.yticks([0, 1, 2, 3])
    plt.tick_params(axis ='y',labelsize =f, length = 8)

  if i[0] == 1 or i[0] == 3:

    plt.yticks([])
    plt.tick_params(axis ='y',labelsize =0, length = 8)

  if i[0] == 0 or i[0] == 1:

    plt.xticks([-3,-2,-1,0,1])
    plt.tick_params(axis ='x',labelsize =0, length = 8)
        
  if i[0] == 2 or i[0] == 3:
    plt.xticks([-3,-2,-1,0,1])
    plt.tick_params(axis ='x',labelsize =f, length = 8)
    plt.xlabel(r"$\log_{\rm 10}(\lambda)$", fontsize = f)

  plot_erd_bin(i[1])


plt.show()