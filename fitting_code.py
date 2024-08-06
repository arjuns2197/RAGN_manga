import math
import numpy as np

from astropy.io import fits
from astropy.utils.data import download_file
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
import scipy.integrate as integrate
import pandas as pd
import scipy
from scipy.stats import norm
from scipy import ndimage
from scipy import interpolate 
from scipy.special import betainc
from scipy.special import gammainc
from scipy.special import gamma
from astropy.table import Column
from astropy.table import Table
from random import random
from scipy.stats import rv_continuous
ln_10 = np.log(10)
log_lambda_max = 1


#correc_obj = agn.correction_interp()
#np.save("../AGN_FIRST_files/data/interp_obj.npy", correc_obj ,allow_pickle= True)
correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]


def mle_prop(model_type,df_manga,correction_integ_obj, use_volw):

  '''Returns --> Model parameters alpha, beta, lamda_star and log_lambda_min after running MLE. 

  Inputs --> 1) A string specifying the model type ("log_Mass" for the mass model) 
  2) A pandas dataframe of properties of all galaxies. This can be generated using "load_matched_df()".
  3) A file containing a grid of interpolations of integrals of the power law correction term that turns the power law into a Schecter function.
  Use the function "correction_interp()" to generate the grid and then save it as a file that can be input here. np.save(, allow_pickle = True) works
  well to save the file.
  4) Set use_volw to True if you want to include the Wake weights while fitting. Set it to False otherwise. 
  
  '''

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

    ln_likelihood = ln_likelihood + (volw_non_agn_alps_gt_1*np.log(likelihoods_alps_gt_1)).sum() + (
    volw_non_agn_alps_lt_1*np.log(likelihoods_alps_lt_1)).sum()

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



def schechter_prop(lamda, prop, alpha_0, beta, lambda_star, property_string):

  """This is derived from the true model. This function is used while integrating the
  model, as it allows us to vary lambda in log units. This is used only in the Monte Carlo stage of
  the analysis"""

  alpha = alpha_prop(alpha_0, prop, beta, property_string)

  
  p_Xi= np.log(10)*lambda_star*(lamda/lambda_star)**(-alpha+1)*np.exp(-lamda/lambda_star)
   
  return p_Xi


def normalisation_prop(prop, alpha_0, beta,
lambda_star,log_lambda_min, property_string):

  """This is the normalisation constant for the schechter_mass() function."""

  integral = integrate.quad(lambda x: schechter_prop(10**x, prop,alpha_0,beta,lambda_star,
  property_string), log_lambda_min,log_lambda_max)[0]

  return 1/integral

v_norma_prop = np.vectorize(normalisation_prop)



def schechter_x(x,alpha):

  """This is used in the likelihood function for the matches. Basically, this is the true model.
  The other Schechter functions are derived from this"""

  f = x**(-alpha)*np.exp(-x)

  return f 


def alpha_prop(alpha_0, prop, beta,property):

  '''This function allows us to parametrise the Schechter function slope alpha as
  a function of various galaxy properties'''

  if property == None:
    prop_0 = 1
    alpha = alpha_0*(prop/prop_0)**beta


  elif property == "log_SFR_SF":
    prop_0 = 6
    alpha = alpha_0*((prop + 8)/prop_0)**beta

  elif property == "log_Mass":
    prop_0 = 10.5
    alpha = alpha_0*(prop/prop_0)**beta

  elif property == "sSFR":
    prop_0 = -11.5
    alpha = alpha_0*(prop/prop_0)**beta

  elif property == "STELLAR_SIGMA_1RE":
    prop_0 = 2.2
    alpha = alpha_0*(prop/prop_0)**beta
    

  if type(alpha) == np.ndarray:

    ind_high_alpha = np.where(alpha >8)

    alpha[ind_high_alpha] = 8

    ind_low_alpha = np.where(alpha < -3)
    alpha[ind_low_alpha] = -3


  return alpha


def power_law_int(log_lam_min,log_lam_max,alpha,lambda_star):

  """Integral of the power law component of the Schechter function"""

  lam_min = 10**log_lam_min
  lam_max = 10**log_lam_max

  f = (lam_max/lambda_star)**(-alpha+1) - (lam_min/lambda_star)**(-alpha+1)

  return lambda_star*f/(-alpha+1)


def correction_func(x,alpha):

  """Correction term"""
  f = np.log(10)*(1- np.exp(-x))*x**(-alpha+1)

  return f


def correction_interp():

  """Creates a 2D interpolation object of the integrals of the correction term, at every log_lam and alpha between some chosen
  range of the two quantities."""
   
  alphas = np.linspace(-3,3.5,100)

  n = 1000
  log_x_lo = -10
  log_x_hi = 6
  d_log_x = (log_x_hi - log_x_lo)/n
  log_x = log_x_lo + d_log_x*(np.arange(n, dtype=np.float64)+0.5)
  log_xs = log_x + 0.5*d_log_x

  alpalp, xx = np.meshgrid(alphas,log_xs)
  phi = correction_func(10**xx,alpalp)
  z = phi.cumsum(axis = 0)*d_log_x

  interp_obj = interpolate.RectBivariateSpline(10**log_xs,alphas,z)

  return interp_obj



def total_integral(log_lam_min,log_lam_max,alpha,lambda_star,correc_interp):
  """Computes the integral of the Schehter function between log_lam_min and log_lam_max, for a specified alpha and lambda_star.
  Splits the integral into a power law term and a correction term before computing the total integral.

  requires a file containing a grid of interpolations of integrals of the correction term. Use the function "correction_interp()" 
  to generate the grid and then save it as a file that can be input here. The function np.save(, allow_pickle = True) works
  well to save the file.
  """

  lam_min = 10**log_lam_min
  lam_max = 10**log_lam_max

  I1 = power_law_int(log_lam_min,log_lam_max,alpha,lambda_star)


  I2 = lambda_star*(correc_interp(lam_max/lambda_star,alpha,
  grid = False)- correc_interp(lam_min/lambda_star,alpha,grid = False))


  return I1 - I2


def gamma_integ(log_min, log_max,alpha, lambda_star):

  """Computes the integral of the Schehter function between log_lam_min and log_lam_max, for a specified alpha and lambda_star. Uses
  the Gamma function and is valid only for alpha < -1.
  """

  f = lambda_star*(gammainc(1-alpha,10**log_max/lambda_star) - gammainc(1-alpha,
  10**log_min/lambda_star))*gamma(1-alpha)

  return f