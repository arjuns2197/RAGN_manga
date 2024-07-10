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

def mle_prop_new_ers(model_type,df_manga,correction_integ_obj, use_volw):

  '''Returns --> Model parameters alpha, beta, lamda_star and log_lambda_min after running MLE. 

  Inputs --> 1) A string specifying the model type ("log_Mass" for the mass model) 
  2) A pandas dataframe of properties of all galaxies. This can be generated using "load_matched_df()".
  3) A file containing a grid of interpolations of integrals of the power law correction term that turns the power law into a Schecter function.
  Use the function "correction_interp()" to generate the grid and then save it as a file that can be input here. np.save(, allow_pickle = True) works
  well to save the file.
  4) Set use_volw to True if you want to include the Wake weights while fitting. Set it to False otherwise. 
  
  '''
  epsilon = 0.1
  c = 3e10

  Lr = df_manga['Lr']
  if type(Lr) != np.ndarray:
    Lr = Lr.to_numpy()


  sfr = df_manga['log_SFR_SF']
  if type(sfr) != np.ndarray:
    sfr = sfr.to_numpy()

  vd_dap = df_manga['STELLAR_SIGMA_1RE']
  if type(vd_dap) != np.ndarray:
    vd_dap = vd_dap.to_numpy()

  z = df_manga['nsa_redshift']
  if type(z) != np.ndarray:
    z = z.to_numpy()

  mass = df_manga['log_Mass']
  if type(mass) != np.ndarray:
    mass = mass.to_numpy()

  if use_volw == True:
    vol_weights = df_manga['ESWEIGHT']
    if type(vol_weights) != np.ndarray:
      vol_weights = vol_weights.to_numpy()
  
  if use_volw == False:
    vol_weights = np.ones(len(df_manga['ESWEIGHT']))
    if type(vol_weights) != np.ndarray:
      vol_weights = vol_weights.to_numpy()

  ind_agn = df_manga["agn flux detectable"].to_numpy(dtype= bool)

  Lr_agn = Lr[ind_agn]
  Lr_non_agn = np.delete(Lr, ind_agn)

  sfr_agn = sfr[ind_agn]
  sfr_non_agn = np.delete(sfr, ind_agn)

  vd_dap_agn = vd_dap[ind_agn]
  vd_dap_non_agn = np.delete(vd_dap, ind_agn)

  z_agn = z[ind_agn]
  z_non_agn = np.delete(z, ind_agn)

  mass_agn = mass[ind_agn]
  mass_non_agn = np.delete(mass, ind_agn)

  ssfr_agn = np.log10(10**sfr_agn/10**mass_agn)
  ssfr_non_agn = np.log10(10**sfr_non_agn/10**mass_non_agn)

  volw_agn = vol_weights[ind_agn]
  volw_non_agn = np.delete(vol_weights, ind_agn)

  lambda_lim_non_agn = get_ER_lim_agn_cav(vd_dap_non_agn,z_non_agn,sfr_non_agn)
  lambda_agn = get_er_pcav(Lr_agn,vd_dap_agn)


  if model_type == None: ## To fit the uniform model. The galaxy property values are just unity.

    prop_agn = np.ones(len(sfr_agn))
    prop_non_agn = np.ones(len(sfr_non_agn))

  if model_type == "log_SFR_SF":

    prop_agn = sfr_agn
    prop_non_agn = sfr_non_agn

  if model_type == "log_Mass":

    prop_agn = mass_agn
    prop_non_agn = mass_non_agn

  if model_type == "STELLAR_SIGMA_1RE":

    prop_agn = vd_dap_agn
    prop_non_agn = vd_dap_non_agn

  if model_type == "sSFR":

    prop_agn = ssfr_agn
    prop_non_agn = ssfr_non_agn
  

  def ln_p_sch(theta): # Likelihood Function

    alpha_0 = theta[0]
    beta = theta[1]
    lambda_star = theta[2]
    log_lambda_min = theta[3]

    alphas_agn = alpha_prop(alpha_0,prop_agn,beta,model_type)

    inds_alp_lt_1 = np.where(alphas_agn < 1)
    inds_alp_gt_1 = np.where(alphas_agn > 1)

    norms_agn = np.ones(len(alphas_agn))

    norms_agn[inds_alp_gt_1] = 1/(total_integral(log_lambda_min,2.2,alphas_agn[inds_alp_gt_1],
    lambda_star,correction_integ_obj)) 
    norms_agn[inds_alp_lt_1] = 1/(gamma_integ(log_lambda_min,2.2,alphas_agn[inds_alp_lt_1],
    lambda_star)) 

    likelihoods_agn = norms_agn*schechter_prop(lambda_agn,
    prop_agn,alpha_0,beta,lambda_star,model_type)

    ln_likelihood = (volw_agn*np.log(likelihoods_agn)).sum()

    alphas_non_agn = alpha_prop(alpha_0,prop_non_agn,beta,model_type)
    norms_non_agn = np.ones(len(alphas_non_agn))

    inds_alps_lt_1 = np.where(alphas_non_agn < 1)
    inds_alps_gt_1 = np.where(alphas_non_agn > 1)

    volw_non_agn_alps_lt_1 = volw_non_agn[inds_alps_lt_1]
    volw_non_agn_alps_gt_1 = volw_non_agn[inds_alps_gt_1]

    norms_non_agn[inds_alps_gt_1] = 1/(total_integral(log_lambda_min,2.2,alphas_non_agn[inds_alps_gt_1],
    lambda_star,correction_integ_obj)) 
    norms_non_agn[inds_alps_lt_1] = 1/(gamma_integ(log_lambda_min,2.2,alphas_non_agn[inds_alps_lt_1],
    lambda_star))  

    likelihoods_alps_gt_1 = norms_non_agn[inds_alps_gt_1]*total_integral(log_lambda_min,
    np.log10(lambda_lim_non_agn[inds_alps_gt_1]),alphas_non_agn[inds_alps_gt_1],lambda_star,correction_integ_obj)

    
    likelihoods_alps_lt_1 = norms_non_agn[inds_alps_lt_1]*gamma_integ(log_lambda_min,
    np.log10(lambda_lim_non_agn[inds_alps_lt_1])
    ,alphas_non_agn[inds_alps_lt_1],lambda_star)

    ln_likelihood = ln_likelihood + (volw_non_agn_alps_gt_1*np.log(
    likelihoods_alps_gt_1)).sum() + (volw_non_agn_alps_lt_1*np.log(likelihoods_alps_lt_1)).sum()

    print(ln_likelihood)

    return -ln_likelihood

  from scipy.optimize import minimize

  alpha0_start = 2
  beta_start = -3
  lambda_star_start = 0.1
  min_start = -6

  theta_start = [alpha0_start,beta_start,lambda_star_start,min_start]
  result = minimize(ln_p_sch,theta_start, method= 'Nelder-Mead',
  bounds = [(None,None),(None,None),(None,None),(None,None)])


  return [result.x[0], result.x[1], result.x[2], result.x[3]]