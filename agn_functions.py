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
import emcee

ln_10 = np.log(10)
log_lambda_max = 1
len_manga = 6731
len_manga_Ha = 6255

#correc_obj = agn.correction_interp()
#np.save("../AGN_FIRST_files/data/interp_obj.npy", correc_obj ,allow_pickle= True)
correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]


#############################################################
#####################################
#############################

### NOTE: Use Pandas version 1.5.3 or lesser

#############################
#####################################
#############################################################

def get_radio(flux,z):

  ''' Returns --> The 1.4 GHz "nuL_nu" radio luminiosities of
  the radio catalog matches in MaNGA.
  
  Inputs --> The radio catalog fluxes of the matches in mJy. MaNGA redshifts
  of the matches'''

  radio_fluxes = 1.4e9*flux*1e-26*u.erg/(u.s*u.cm*u.cm*u.Hz)
  distance_L= cosmo.luminosity_distance(z).cgs  
  radio_L = radio_fluxes*(4*np.pi*distance_L*distance_L) # In erg/s/Hz.
  radio_L= np.log10(radio_L.value)
  return radio_L

##########################################################

def suppression_factor(Lr, Lc):

  '''
  Returns --> Suppression factor "f".
  Inputs --> Radio Luminosity 'Lr' and 
  the Hopkins characteristic radio luminosity 'Lc'
  
  Refer to 2003ApJ...599..971H for further information.
  '''

  f = (0.1 + 0.9 * (10**Lr/10**Lc)**0.3)**-1

  return np.log10(f)


def hopkins_sfr(Lr, Lc, intercept, dex_difference_imf):
    
  '''
  Returns --> Star formation rate according to the Hopkins relation (Can include a dex difference in SFR to
  account for the usage of IMFs different than Salpeter. Set dex_difference to 0 for Salpeter).
  Inputs --> The radio luminosity at which SFR is required. Lc and the log space y-intercept of the
  Hopkins L-SFR relation'''

  sfr = Lr - intercept - dex_difference_imf


  if Lr < 37.95:

    sfr = suppression_factor(Lr, Lc) + sfr

  return sfr

v_hopkins_sfr = np.vectorize(hopkins_sfr)

Lc = 37.95 ## Characteristic luminosity from Hopkins.
Lumi_r = np.linspace(1,50,1000)

Lsfr_interp = scipy.interpolate.interp1d(v_hopkins_sfr(Lumi_r,Lc,37.4,0.25), Lumi_r) # 0.25 for Chabrier IMF
Lsfr_interp_hopkins = scipy.interpolate.interp1d(v_hopkins_sfr(Lumi_r,Lc,37.4,0), Lumi_r)

##########################################################

def load_drpall_data():

  path_drpall = "../AGN_FIRST_files/data/drpall-v3_1_1.fits"
  data_drpall = fits.open(path_drpall)[1].data

  print("Initial length:", len(data_drpall))

  mngtarg3 = data_drpall['MNGTARG3']
  
  inds_bad_objs = []

  for i in range(11273):

    if np.log2(mngtarg3[i]) in [8,9,18,19,20,21,22,27]:
      
      inds_bad_objs.append(i)


  data_drpall = np.delete(data_drpall, inds_bad_objs)

  print("Final length: ", len(data_drpall))
  
  return data_drpall

def match_manga_first(ra_manga,dec_manga,ra_first,dec_first,max_sep):

  c_manga = SkyCoord(ra = ra_manga*u.degree, dec= dec_manga*u.degree, 
  frame= 'icrs', unit='deg')
  c_first = SkyCoord(ra = ra_first*u.degree,dec= dec_first*u.degree, frame= 'icrs', unit='deg') 
  idx, sep2d, d3d = c_manga.match_to_catalog_sky(c_first)
  sep_constraint = sep2d < max_sep*u.arcsecond
  inds_matches_manga = np.arange(len(ra_manga))[sep_constraint]
  inds_matches_first = np.arange(len(ra_first))[idx[sep_constraint]]

  return [inds_matches_manga,inds_matches_first]

def get_integ_nvss_flux(theta_M,theta_m,theta_N,peak):

  """Input theta_M, theta_m in degrees. Theta_N in arcsecs. peak in Jy"""

  theta_M = theta_M*3600
  theta_m = theta_m*3600
  peak = peak*1000

  fint = peak*(theta_M*theta_m/(theta_N**2))

  return fint


def load_drpall_df():

  ''' '''

  # load data
  data_drpall = load_drpall_data()

  #url_nvss = "https://urldefense.proofpoint.com/v2/url?u=https-3A__www.dropbox.com_s_lyuy4lp2lzi5p4d_CATALOG.FIT-3Fdl-3D0&d=DwMFaQ&c=slrrB7dE8n7gBJbeO0g-IQ&r=J9TPhCTsXLU5rvH37Iir6w&m=iqahptD4Y5INWgEYix4rQqPS-qYAZmwpc7Kj0bS2Que1p58myO0ZPWeMXhMgupDZ&s=TbiwgLzBNf8Rusy5ojMbZ8Gjhsc7DWPc0gOhJZvJasw&e="
  path_nvss = '../AGN_FIRST_files/data/nvss_full_catalog.FIT' 
  data_nvss = fits.open(path_nvss,memmap=True)[1].data 

  fints = get_integ_nvss_flux(data_nvss['MAJOR AX'], data_nvss['MINOR AX'], 45, data_nvss['PEAK INT'])
  data_nvss = Table(data_nvss)
  data_nvss['FINT'] = fints

  #url_first = 'https://drive.google.com/uc?export=download&id=1itAIFo_i30j9kFxOGxIPDuxuvIJXyUNm'
  path_first = '../AGN_FIRST_files/data/FIRST.fits' 
  data_first = fits.open(path_first,memmap=True)[1].data

  c_nvss = SkyCoord(ra=data_nvss['RA(2000)']*u.degree, dec= data_nvss['DEC(2000)']*u.degree,
  frame= 'icrs', unit='deg')
  c_drpall = SkyCoord(ra=data_drpall['OBJRA']*u.degree,dec= data_drpall['OBJDEC']*u.degree,
  frame= 'icrs', unit='deg') 
  c_first = SkyCoord(ra=data_first['RA']*u.degree, dec= data_first['DEC']*u.degree,
  frame= 'icrs', unit='deg')

  #Match catalogs upto 3 arcmins
  idx_nvss, idx_drpall,d2d, d3d = c_drpall.search_around_sky(c_nvss, 3*u.arcmin)
  unique, counts = np.unique(idx_drpall, return_counts=True)  

  len_drpall = len(data_drpall['PLATEIFU'])
 
  num_nvss_matches3m = np.zeros(len_drpall)
  num_nvss_matches3m[unique] = counts

  nvss_match_inds3m = np.zeros(len_drpall, dtype = 'object')
  nvss_match_inds3m = get_matching_inds(len_drpall,idx_drpall,idx_nvss)
  

  ##FIRST 3 arcmins

  idx_first, idx_drpall_first,d2d_first, d3d_first = c_drpall.search_around_sky(c_first,
  3*u.arcmin)
  unique_first, counts_first = np.unique(idx_drpall_first, return_counts=True)  
  num_first_matches3m = np.zeros(len_drpall)
  num_first_matches3m[unique_first] = counts_first

  first_match_inds3m = np.zeros(len_drpall, dtype = 'object')
  first_match_inds3m = get_matching_inds(len_drpall,idx_drpall_first,
  idx_first)


  # FIRST 30 arcsec

  idx_first_30s,idx_drpall_first_30s,d2d_first_30s,d3d_first_30s = c_drpall.search_around_sky(
  c_first,
  30*u.arcsec)
  unique_first_30s, counts_first_30s = np.unique(idx_drpall_first_30s, return_counts=True)  
  num_first_matches30s = np.zeros(len_drpall)
  num_first_matches30s[unique_first_30s] = counts_first_30s

  first_match_inds30s = np.zeros(len_drpall, dtype = 'object')
  first_match_inds30s = get_matching_inds(len_drpall,idx_drpall_first_30s,
  idx_first_30s)


  # Get RA and DEC of 3 arcmin matches
  ra_first3m = get_property_at_indices(data_first,first_match_inds3m,'RA')
  dec_first3m = get_property_at_indices(data_first,first_match_inds3m,'DEC')

  ra_first30s = get_property_at_indices(data_first,first_match_inds30s,'RA')
  dec_first30s = get_property_at_indices(data_first,first_match_inds30s,'DEC')

  ra_nvss3m = get_property_at_indices(data_nvss,nvss_match_inds3m,'RA(2000)')
  dec_nvss3m = get_property_at_indices(data_nvss,nvss_match_inds3m,'DEC(2000)')

  flux_first3m = get_property_at_indices(data_first,first_match_inds3m,'FINT')
  flux_first30s = get_property_at_indices(data_first,first_match_inds30s,'FINT')
  flux_nvss3m = get_property_at_indices(data_nvss,nvss_match_inds3m,'FINT')

  maj_ax_first30s = get_property_at_indices(data_first,first_match_inds30s,'MAJOR')
  pos_ang_first30s = get_property_at_indices(data_first,first_match_inds30s,'POSANG')

  z = data_drpall['z']
  ra = data_drpall["OBJRA"]
  dec = data_drpall["OBJDEC"]
  plateifu = data_drpall['PLATEIFU']

  ra_first = data_first['RA']
  dec_first = data_first['DEC']

  ## To find matches with MaNGA in FIRST within 3 arcsec
  [inds_manga, inds_first] = match_manga_first(ra,dec,ra_first,dec_first,3)

  has_3arcsec_first_match = np.zeros(len(ra))
  has_3arcsec_first_match[inds_manga] = np.ones(len(inds_manga))


  dict_drpall = {"plateifu":plateifu,"RA": ra, "DEC": dec, "No. NVSS matches(3m)" : num_nvss_matches3m,
  "Indices NVSS matches(3m)" : nvss_match_inds3m,
  "3 arcsec FIRST match": has_3arcsec_first_match,
  "No. FIRST matches(3m)" : num_first_matches3m,
  "Indices FIRST matches(3m)": first_match_inds3m,"RA 3)":ra_first3m,
  "DEC FIRST(3m)": dec_first3m,"RA NVSS(3m)":ra_nvss3m,
  "DEC NVSS(3m)":dec_nvss3m,"RA FIRST(30s)":ra_first30s,
  "DEC FIRST(30s)":dec_first30s,
  "Flux FIRST(3m)":flux_first3m,
  "Flux FIRST(30s)":flux_first30s,"Flux NVSS(3m)":flux_nvss3m,
  "No. FIRST matches(30s)": num_first_matches30s,"Maj FIRST(30s)": maj_ax_first30s,
  "PA FIRST(30s)":pos_ang_first30s, "Indices FIRST matches(30s)": first_match_inds30s,
  "nsa_redshift":z}

  df_drpall = pd.DataFrame(dict_drpall)


  return df_drpall


def delete_unrelated_flux_2_1(df_matched,index):

  if is_candidate_single_two_matches(df_matched,index) == True:
    required_index = is_candidate_double(df_matched,index)[1]

    df_matched["Flux NVSS(3m)"].iloc[index] = [df_matched["Flux NVSS(3m)"].iloc[index][required_index]]
    df_matched["RA NVSS(3m)"].iloc[index] = [df_matched["RA NVSS(3m)"].iloc[index][required_index]]
    df_matched["DEC NVSS(3m)"].iloc[index] = [df_matched["DEC NVSS(3m)"].iloc[index][required_index]]
    df_matched["Indices NVSS matches(3m)"].iloc[index] = [df_matched["Indices NVSS matches(3m)"].iloc[index][required_index]]
    df_matched["No. NVSS matches(3m)"].iloc[index] = 1

    if "Flux Error NVSS(3m)" in df_matched.columns:
      df_matched["Flux Error NVSS(3m)"].iloc[index] = [df_matched["Flux Error NVSS(3m)"].iloc[index][required_index]]

def get_property_at_indices(data,indices,property):

  if property == 'FINT':
    property_at_indices = []
    for i in indices:
      if i == 0:
        property_at_indices.append(0)

      else:
        property_at_indices.append(np.array(data[property][i]))
    
  else:
        
    property_at_indices = []
    for i in indices:

      if i == 0:
        property_at_indices.append(None)
    
      else:
        property_at_indices.append(data[property][i])
 

  return np.array(property_at_indices, dtype= 'object')
  
 

def get_matching_inds(data_length,idx1,idx2):

  '''
  Returns --> An array containing lists of indices of matches in idx2 for each unique
  element in idx1. The size of the array will be equal to the number of unique elements
  in idx1. '''
       
  matches = np.zeros(data_length, dtype = 'object')
    
  for i in range(len(idx1)):
        
    if matches[idx1[i]] != 0:
      matches[idx1[i]].append(idx2[i])
        
    if matches[idx1[i]] == 0:
      matches[idx1[i]] = [idx2[i]]
            
  return matches

###################################################################

def is_agn(sfr,Lr):

  '''
  Returns --> Either True or False for a galaxy's AGN-ness
  Inputs --> The SFR and radio luminosity of the galaxy whose AGN-ness
  is to be determined.
  '''

  if sfr > 0.044:

    if Lr > Lsfr_interp(sfr) + 1:
      return True

    else:
      return False

  elif Lr > 38.6:

    return True

  else:

    return False

vector_is_agn = np.vectorize(is_agn)


def flux_detected(sfr,Lr,z,check_agn):

  '''Returns an array of True values where the input radio galaxies with a given sfr, Lr and z may be detected at a specified flux limit
  If check_agn is set to True, the function checks for an agn detection. If it is set to False, the detection
  might be an agn or might be due to star formation'''

  flux_lim = 2.5
  
  if check_agn == True:
    
    if vector_is_agn(sfr,Lr):

      flux = get_flux(Lr,z)

      if flux > flux_lim:
      
        return True
      
      else:
        return False

    else:
      return False

  if check_agn == False:

    flux_1 = get_flux(Lr,z)

    if flux_1 > flux_lim:
      return True

    else: 
      return False

  else:
    return False

vector_flux_detected = np.vectorize(flux_detected)


def is_agn_plain_cut(sfr,Lr):

  '''
  Returns --> Either True or False for a galaxy's AGN-ness
  Inputs --> The SFR and radio luminosity of the galaxy whose AGN-ness
  is to be determined.
  '''

  if Lr > Lsfr_interp(sfr) + 1:
    return True

  else:
    return False


vector_is_agn_plain_cut = np.vectorize(is_agn_plain_cut)

def flux_detected_plain_cut(sfr,Lr,z,check_agn):

  '''Returns an array of True values where the input radio galaxies with a given sfr, Lr and z may be detected at a specified flux limit
  If check_agn is set to True, the function checks for an agn detection. If it is set to False, the detection
  might be an agn or might be due to star formation'''

  flux_lim = 2.5
  
  if check_agn == True:
    
    if vector_is_agn_plain_cut(sfr,Lr):

      flux = get_flux(Lr,z)

      if flux > flux_lim:
      
        return True
      
      else:
        return False

    else:
      return False

  if check_agn == False:

    flux_1 = get_flux(Lr,z)

    if flux_1 > flux_lim:
      return True

    else: 
      return False

  else:
    return False

vector_flux_detected_plain_cut = np.vectorize(flux_detected_plain_cut)

#########################################################################


def get_ER(vd,radio_l): 
  
  """  Returns the Eddington Ratio of the detected AGN. Inputs the velocity dispersion and radio luminosity of the galaxy."""
  
  log_mass_BH = np.float64(8.12+4.24*np.log10(10**vd/200))  # log(M_bh/Msun)
  log_L_Edd= np.float64(np.log10(1.26e38)+log_mass_BH)
  L_Edd= 10**log_L_Edd
  log_L_XR = np.float64(0.925*radio_l+7.1)
  L_XR= 10**log_L_XR
  L_Bolo= 20*(L_XR)
  lambda_ = L_Bolo/L_Edd

  return lambda_

def get_flux(radio_l,z):
  '''Returns measured flux from the input log radio luminosity and redshift'''

  radio_l = 10**radio_l
  distance_L = cosmo.luminosity_distance(z).cgs.value 
  radio_flux = radio_l/(4*np.pi*distance_L*distance_L)
  flux_in_mjy = radio_flux/(1.4e9*1e-26)

  return flux_in_mjy


def get_L_lim(z):

  """Returns --> The log of the limiting luminosity of a radio source in MaNGA galaxies
  without matches in the radio catalog. 

  Inputs --> redshift and the flux threshold of the radio telescope in mJy"""
 
  flux_lim = 2.5

  distance_lumi = cosmo.luminosity_distance(z).cgs
  L_lim1 = np.log10(4*np.pi*(distance_lumi.value*distance_lumi.value)*flux_lim*1e-26*1.4e9)

  return L_lim1


def get_ER_lim_z(vd,z):
  
  """  Returns the limiting AGN Eddington Ratio of MaNGA galaxies without radio detection. 
  Inputs the velocity dispersion and redshift of the galaxies."""

  lumi_lim = get_L_lim(z)

  log_mass_BH_all = np.float64(8.12+4.24*np.log10(10**vd/200)) 
  log_L_Edd_all = np.float64(np.log10(1.26e38)+log_mass_BH_all)
  L_Edd_all= 10**log_L_Edd_all
  log_L_XR_all = np.float64(0.925*lumi_lim+7.1)
  L_XR_all = 10**log_L_XR_all
  L_Bolo_all = 20*L_XR_all
  lambda_lim = L_Bolo_all/L_Edd_all

  return lambda_lim


def get_ER_lim_agn(vd,z,sfr):
  
  """  Returns the limiting AGN Eddington Ratio of MaNGA galaxies without radio detection. 
  Inputs the velocity dispersion and redshift of the galaxies."""

  lumi_lim = get_L_lim(z)
  lumi_lim = v_get_AGN_lim(sfr,lumi_lim)

  log_mass_BH_all = np.float64(8.12+4.24*np.log10(10**vd/200)) 
  log_L_Edd_all = np.float64(np.log10(1.26e38)+log_mass_BH_all)
  L_Edd_all= 10**log_L_Edd_all
  log_L_XR_all = np.float64(0.925*lumi_lim+7.1)
  L_XR_all = 10**log_L_XR_all
  L_Bolo_all = 20*L_XR_all
  lambda_lim = L_Bolo_all/L_Edd_all

  return lambda_lim


def get_AGN_lim(sf,L_lim):

  """Returns the limiting luminosity of an AGN in the MaNGA galaxies
  without matches in the radio catalog."""

  SF_radio_lim =  Lsfr_interp(sf) + 1 # We're providing a buffer of 1 dex above the regressed line before classifying a galaxy as an AGN.
                                          
  L_lim_modified = L_lim
  
  if sf < 0.044:
    if 38.6 > L_lim_modified: # used to be 38.6
      L_lim_modified = 38.6

  else:
    if SF_radio_lim > L_lim_modified:
      L_lim_modified = SF_radio_lim

  return L_lim_modified

v_get_AGN_lim = np.vectorize(get_AGN_lim)

#####################################################



#####################################################
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

  total_integral = I1 - I2
  
  return total_integral


def gamma_integ(log_min, log_max,alpha, lambda_star):

  """Computes the integral of the Schehter function between log_lam_min and log_lam_max, for a specified alpha and lambda_star. Uses
  the Gamma function and is valid only for alpha < -1.
  """

  f = lambda_star*(gammainc(1-alpha,10**log_max/lambda_star) - gammainc(1-alpha,
  10**log_min/lambda_star))*gamma(1-alpha)

  return f

##################################################
#Identification of NVSS doubles

def fwmp_2(ra1,dec1,ra2,dec2,f1,f2):
    
  '''
  Returns -> Flux weighted mean position of two radio sources
  Inputs -> The RAs and DECs of the two sources and their fluxes. 
  RAs and DECs should be in degrees and the fluxes should be in units of mJy
  '''

  num_ra = f1*ra1 + f2*ra2
  denom_ra = (f1+f2)

  num_dec = f1*dec1 + f2*dec2
  denom_dec = (f1+f2)

  return [num_ra/denom_ra, num_dec/denom_dec]


def is_candidate_double(df_manga,index):

  '''
  Returns -> True if a given galaxy in the MaNGA catalog could potentially
  be associated with two NVSS sources. False otherwise.

  Inputs -> The MaNGA catalog dataframe and index of the galaxy to be evaluated
  '''

  if df_manga['No. NVSS matches(3m)'][index] !=2:
    return False
  
  ra1 = df_manga['RA NVSS(3m)'][index][0]
  dec1 = df_manga['DEC NVSS(3m)'][index][0]

  ra2 = df_manga['RA NVSS(3m)'][index][1]
  dec2 = df_manga['DEC NVSS(3m)'][index][1]

  flux_nvss1 = df_manga['Flux NVSS(3m)'][index][0]
  flux_nvss2 = df_manga['Flux NVSS(3m)'][index][1]
        
  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]


  c1 = SkyCoord(ra1, dec1, frame='icrs', unit = 'degree')

  c2 = SkyCoord(ra2, dec2, frame='icrs', unit = 'degree')

  c_manga = SkyCoord(ra_manga, dec_manga, frame='icrs', unit = 'degree')

  sep1 = c_manga.separation(c1).value*3600
  sep2 = c_manga.separation(c2).value*3600

  if sep1 < 15 and sep2 > 15: 
    return ["Candidate Single", 0]

  if sep2 < 15 and sep1 > 15:
    return ["Candidate Single", 1]
  
  if sep1 < 90 and sep2 < 90:

    [ra_fwmp, dec_fwmp] = fwmp_2(ra1,dec1,ra2,dec2,flux_nvss1,
    flux_nvss2)

    c_fwmp = SkyCoord(ra_fwmp, dec_fwmp,frame = 'icrs', unit = 'degree')
    sep_fwmp = c_manga.separation(c_fwmp).value*3600

      

    if sep_fwmp < 15:

      sep_near = min(sep1,sep2)
      sep_far = max(sep1,sep2)
      critical_separation = min(15,np.abs((sep_far-20)))
          

      if sep_near > critical_separation:

        return True
      
      else:
        return False
      
    else:
        return False

  else:
    return False

  
    
def get_agn_angle(ra1,dec1,ra2,dec2,ra_cent,dec_cent):

  '''
  Returns -> Angle in degrees of the line joining radio source 1, a central galaxy and
  radio source 2

  Inputs -> RAs and DECs of the radio sources and the central galaxy
  '''
    
  vec1_x = ra1 - ra_cent
  vec1_y = dec1 - dec_cent
  mag_vec1 = (vec1_x**2 + vec1_y**2)**0.5
  
  vec2_x = ra2 - ra_cent
  vec2_y = dec2 - dec_cent
  mag_vec2 = (vec2_x**2 + vec2_y**2)**0.5
  
  angle_double = np.arccos((vec1_x*vec2_x + vec1_y*vec2_y)/(mag_vec1*mag_vec2))
  angle_in_degrees = angle_double*180/np.pi
  
  return angle_in_degrees



def is_nvss_double(df_manga,index):

  '''
  Returns -> 0 if no radio source is detected. 4 if an NVSS double radio
  source is found to be associated with a given MaNGA galaxy. "Needs VI" if
  it is decided that the given galaxy needs to be visually inspected before
  classyfying it as a NVSS double source

  Inputs -> MaNGA dataframe and the index of the galaxy to be evaluated
  '''

  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]
  c_manga = SkyCoord(ra_manga,dec_manga,frame='icrs',unit = 'degree')


  if df_manga['No. NVSS matches(3m)'][index] !=2:
    return 0
  
  if is_candidate_double(df_manga,index) != True:
    return 0

  if df_manga['3 arcsec FIRST match'][index] == 1:
    return 4
  
  if df_manga['No. FIRST matches(30s)'][index] == 0:

    ra1 = df_manga['RA NVSS(3m)'][index][0]
    dec1 = df_manga['DEC NVSS(3m)'][index][0]
    
    ra2 = df_manga['RA NVSS(3m)'][index][1]
    dec2 = df_manga['DEC NVSS(3m)'][index][1]

    c1 = SkyCoord(ra1,dec1,frame='icrs',unit = 'degree')
    c2 = SkyCoord(ra2,dec2,frame='icrs',unit = 'degree')

    sep1 = c_manga.separation(c1).value*3600
    sep2 = c_manga.separation(c2).value*3600

    if sep1 < 60 and sep2 < 60:

      if get_agn_angle(ra1,dec1,ra2,dec2,ra_manga,dec_manga) > 135:

        return 4
          
      else:
        return "Needs VI 4(2)"

    else:
      return "Needs VI 4(2)"
          
  elif (df_manga["No. FIRST matches(30s)"][index] <= 3) and (df_manga["No. FIRST matches(30s)"][index] > 0):
      
    seps = []

    for i in range(len(df_manga['RA FIRST(30s)'][index])):

      ra_i = df_manga['RA FIRST(30s)'][index][i]
      dec_i = df_manga['DEC FIRST(30s)'][index][i]

      c_i = SkyCoord(ra_i, dec_i, frame = 'icrs', unit = 'degree')

      sep = c_manga.separation(c_i).value*3600
      seps.append(sep)
    seps = np.array(seps)

    if (seps > 15).all():
      if (df_manga['Flux FIRST(30s)'][index].sum() > 0.5*df_manga['Flux NVSS(3m)'][index].sum()):
        return 0
          
      else:
        return "Needs VI 4(2)"
      
    else:
      return "Needs VI 4(2)"
  
  else:
    return "Needs VI 4(2)"
    
###################################################

#To identify NVSS Triples

def fwmp_3(ra1,dec1,ra2,dec2,ra3,dec3,f1,f2,f3):

  '''
  Returns -> Flux weighted mean position of 3 radio sources

  Inputs -> RAs, DECs and the fluxes of the 3 sources
  '''
  
  num_ra = f1*ra1 + f2* ra2 + f3*ra3
  denom_ra = f1 + f2 + f3

  num_dec = f1*dec1 + f2*dec2 + f3*dec3
  denom_dec = f1 + f2 + f3

  return [num_ra/denom_ra, num_dec/denom_dec]

  
def is_candidate_triple(df_manga, index):

  if df_manga['No. NVSS matches(3m)'][index] !=3:
    return False
  
  ra1 = df_manga['RA NVSS(3m)'][index][0]
  dec1 = df_manga['DEC NVSS(3m)'][index][0]

  ra2 = df_manga['RA NVSS(3m)'][index][1]
  dec2 = df_manga['DEC NVSS(3m)'][index][1]

  ra3 = df_manga['RA NVSS(3m)'][index][2]
  dec3 = df_manga['DEC NVSS(3m)'][index][2]

  flux_nvss1 = df_manga['Flux NVSS(3m)'][index][0]
  flux_nvss2 = df_manga['Flux NVSS(3m)'][index][1]
  flux_nvss3 = df_manga['Flux NVSS(3m)'][index][2]
        
  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]

  c1 = SkyCoord(ra1, dec1, frame='icrs', unit = 'degree')
  c2 = SkyCoord(ra2, dec2, frame='icrs', unit = 'degree')
  c3 = SkyCoord(ra3, dec3, frame='icrs', unit = 'degree')

  c_manga = SkyCoord(ra_manga, dec_manga, frame='icrs', unit = 'degree')

  sep1 = c_manga.separation(c1).value*3600
  sep2 = c_manga.separation(c2).value*3600
  sep3 = c_manga.separation(c3).value*3600  

  
  if (sep1 < 120) and (sep2 < 120) and (sep3 < 120):
    

    [ra_fwmp, dec_fwmp] = fwmp_3(ra1,dec1,ra2,dec2,ra3,dec3,flux_nvss1,flux_nvss2,flux_nvss3)
    c_fwmp = SkyCoord(ra_fwmp, dec_fwmp, frame = 'icrs', unit = 'degree')
    sep_fwmp = c_manga.separation(c_fwmp).value*3600
    
    
    if sep_fwmp < 15:
      
      return True
    
    
    seps_array = np.array([sep1, sep2, sep3])
    inds_sort = seps_array.argsort()
    seps_array = seps_array[inds_sort]

    ras = np.array([ra1,ra2,ra3])[inds_sort]
    decs = np.array([dec1, dec2, dec3])[inds_sort]
    fluxs = np.array([flux_nvss1, flux_nvss2, flux_nvss3])[inds_sort]

    ra_near = ras[0]
    ra_mid = ras[1]
    ra_far = ras[2]

    dec_near = decs[0]
    dec_mid = decs[1]
    dec_far = decs[2]

    f_near = fluxs[0]
    f_mid = fluxs[1]
    f_far = fluxs[2]

    [ra_fwmp_far2, dec_fwmp_far2] = fwmp_2(ra_mid,dec_mid,ra_far,dec_far,f_mid,f_far)
    c_fwmp_far2 = SkyCoord(ra_fwmp_far2, dec_fwmp_far2,frame = 'icrs',unit = 'degree')
    sep_fwmp_far2 = c_manga.separation(c_fwmp_far2).value*3600


    if sep_fwmp_far2 < 15:
      return True
    
    sep_near = seps_array[0]
    sep_mid = seps_array[1]
    sep_far = seps_array[2]

    if sep_near < 15:
      
      if get_agn_angle(ra_mid,dec_mid, ra_far,dec_far,ra_near,dec_near) > 135:
      
        if sep_mid < 90 and sep_far < 90:
      
          return True
        
        else:
          return False
      
      else:
        return False
    else:
      return False
    

  else:
    return False
  


def is_nvss_triple(df_manga,index):

  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]
  c_manga = SkyCoord(ra_manga,dec_manga,frame='icrs',unit = 'degree')

  if df_manga['No. NVSS matches(3m)'][index] !=3:
    return 0
  
  if is_candidate_triple(df_manga,index) == False:
    return 0

  if df_manga['3 arcsec FIRST match'][index] == 1:
    return 4
  
  if (df_manga['No. FIRST matches(30s)'][index] <= 3) and (df_manga['No. FIRST matches(30s)'][index] > 0):
    
    seps = []

    for i in range(len(df_manga['RA FIRST(30s)'][index])):

      ra_i = df_manga['RA FIRST(30s)'][index][i]
      dec_i = df_manga['DEC FIRST(30s)'][index][i]

      c_i = SkyCoord(ra_i, dec_i, frame = 'icrs', unit = 'degree')

      sep = c_manga.separation(c_i).value*3600
      seps.append(sep)
      
    seps = np.array(seps)

    if (seps > 15).all():

      if (df_manga['Flux FIRST(30s)'][index].sum() > 0.5*df_manga['Flux NVSS(3m)'][index].sum()):
        return 0
      else:
        return "Needs VI 4(3)"
        
    else:
        return "Needs VI 4(3)"

  else:
    return "Needs VI 4(3)"
  

def is_candidate_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2):

  '''
  Returns -> True if any two of the 3 radio sources in the vicinity of a given
  galaxy might potentially be an NVSS double

  Inputs -> MaNGA dataframe, index of the galaxy to be evaluated, the positions
  and fluxes of the three radio sources
  
  '''

  if df_manga['No. NVSS matches(3m)'][index] !=3:
    return False
     
  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]


  c1 = SkyCoord(ra1, dec1, frame='icrs', unit = 'degree')

  c2 = SkyCoord(ra2, dec2, frame='icrs', unit = 'degree')

  c_manga = SkyCoord(ra_manga, dec_manga, frame='icrs', unit = 'degree')

  sep1 = c_manga.separation(c1).value*3600
  sep2 = c_manga.separation(c2).value*3600

  if sep1 < 15 and sep2 > 15: 
    return ["Candidate Single",0]

  if sep2 < 15 and sep1 > 15:
    return ["Candidate Single",1]      
  
  if sep1 < 90 and sep2 < 90:

    [ra_fwmp, dec_fwmp] = fwmp_2(ra1,dec1,ra2,dec2,f1,
    f2)

    c_fwmp = SkyCoord(ra_fwmp, dec_fwmp,frame = 'icrs', unit = 'degree')
    sep_fwmp = c_manga.separation(c_fwmp).value*3600

      

    if sep_fwmp < 15:

      sep_near = min(sep1,sep2)
      sep_far = max(sep1,sep2)
      critical_separation = min(15,np.abs((sep_far-20)))
      

      if sep_near > critical_separation:

        return True
          
      else:
        return False
  
    else:
        return False

  else:
    return False
  

def is_nvss_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2):


  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]
  c_manga = SkyCoord(ra_manga,dec_manga,frame='icrs',unit = 'degree')


  if df_manga['No. NVSS matches(3m)'][index] !=3:
    return 0
  
  elif is_candidate_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2) == False:
    return 0

  elif df_manga['3 arcsec FIRST match'][index] == 1:

    return 4
  
  elif df_manga['No. FIRST matches(30s)'][index] == 0:

    c1 = SkyCoord(ra1,dec1,frame='icrs',unit = 'degree')
    c2 = SkyCoord(ra2,dec2,frame='icrs',unit = 'degree')

    sep1 = c_manga.separation(c1).value*3600
    sep2 = c_manga.separation(c2).value*3600

    if sep1 < 60 and sep2 < 60:

      if get_agn_angle(ra1,dec1,ra2,dec2,ra_manga,dec_manga) > 135:

        return 4
          
      else:
        return "Needs VI 4(2)"

    else:
      return "Needs VI 4(2)"
          
  elif (df_manga["No. FIRST matches(30s)"][index] <= 3) and (df_manga["No. FIRST matches(30s)"][index] > 0):
      
    seps = []

    for i in range(len(df_manga['RA FIRST(30s)'][index])):

      ra_i = df_manga['RA FIRST(30s)'][index][i]
      dec_i = df_manga['DEC FIRST(30s)'][index][i]

      c_i = SkyCoord(ra_i, dec_i, frame = 'icrs', unit = 'degree')

      sep = c_manga.separation(c_i).value*3600
      seps.append(sep)

    seps = np.array(seps)
    if (seps > 15).all():
      if (df_manga['Flux FIRST(30s)'][index].sum() > 0.5*df_manga['Flux NVSS(3m)'][index].sum()):
        return 0
          
      else:
        return "Needs VI 4(2)"
      
    else:
      return "Needs VI 4(2)"
  
  else:
    return "Needs VI 4(2)"


def indices_double_3matches(df_manga, index):

  if df_manga['No. NVSS matches(3m)'][index] != 3:
    return 0

  if is_nvss_triple(df_manga,index) != False:
    return 0
  
  #1st and 2nd galaxies out of the three matches
  ra1 = df_manga['RA NVSS(3m)'][index][0]
  dec1 = df_manga['DEC NVSS(3m)'][index][0]

  ra2 = df_manga['RA NVSS(3m)'][index][1]
  dec2 = df_manga['DEC NVSS(3m)'][index][1]

  f1 = df_manga['Flux NVSS(3m)'][index][0]
  f2 = df_manga['Flux NVSS(3m)'][index][1]    

  if is_nvss_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2) == True:
    return [0,1]
  
  if type(is_candidate_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2)) == list:
    required_index = is_candidate_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2)[1]
    return ("Candidate Single", [0,1][required_index])
     
  #2nd and 3rd galaxies out of the three matches
  ra1 = df_manga['RA NVSS(3m)'][index][1]
  dec1 = df_manga['DEC NVSS(3m)'][index][1]

  ra2 = df_manga['RA NVSS(3m)'][index][2]
  dec2 = df_manga['DEC NVSS(3m)'][index][2]

  f1 = df_manga['Flux NVSS(3m)'][index][1]
  f2 = df_manga['Flux NVSS(3m)'][index][2]    

  if is_nvss_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2) == True:
    return [1,2]
  
  if type(is_candidate_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2)) == list:

    required_index = is_candidate_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2)[1]

    return ("Candidate Single", [1,2][required_index])
  
  #1st and 3rd galaxies out of the three matches
  ra1 = df_manga['RA NVSS(3m)'][index][0]
  dec1 = df_manga['DEC NVSS(3m)'][index][0]

  ra2 = df_manga['RA NVSS(3m)'][index][2]
  dec2 = df_manga['DEC NVSS(3m)'][index][2]

  f1 = df_manga['Flux NVSS(3m)'][index][0]
  f2 = df_manga['Flux NVSS(3m)'][index][2]    

  if is_nvss_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2) == True:
    return [0,2]

  if type(is_candidate_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2)) == list:

    required_index = is_candidate_double_3matches(df_manga,index,ra1,dec1,ra2,dec2,f1,f2)[1]
    return ("Candidate Single", [0,2][required_index])
  
  else:
    return 0
  

def is_candidate_single_three_matches(df_manga,index):

  if type(indices_double_3matches(df_manga,index)) == tuple:
    return True
  
  else:
    return False


def delete_unrelated_flux_3_1(df_manga,index):
  if is_candidate_single_three_matches(df_manga,index) == True:
    required_index = indices_double_3matches(df_manga,index)[1]

    df_manga["Flux NVSS(3m)"].iloc[index] = [df_manga["Flux NVSS(3m)"].iloc[index][required_index]]
    df_manga["RA NVSS(3m)"].iloc[index] = [df_manga["RA NVSS(3m)"].iloc[index][required_index]]
    df_manga["DEC NVSS(3m)"].iloc[index] = [df_manga["DEC NVSS(3m)"].iloc[index][required_index]]
    df_manga["Indices NVSS matches(3m)"].iloc[index] = [df_manga["Indices NVSS matches(3m)"].iloc[index][required_index]]
    df_manga["No. NVSS matches(3m)"].iloc[index] = 1

    if "Flux Error NVSS(3m)" in df_manga.columns:
      df_manga["Flux Error NVSS(3m)"].iloc[index] = [df_manga["Flux Error NVSS(3m)"].iloc[index][required_index]]


  ###########################################

  #Quad NVSS sources

def fwmp_4(ra1,dec1,ra2,dec2,ra3,dec3,ra4,dec4,f1,f2,f3,f4):

  ra_num = ra1*f1 + ra2*f2 + ra3*f3 + ra4*f4
  ra_denom = f1+ f2+ f3+ f4

  dec_num = dec1*f1+ dec2*f2+ dec3*f3+ dec4*f4
  dec_denom = f1 + f2 + f3 + f4

  return [ra_num/ra_denom, dec_num/dec_denom]
  

def is_candidate_quad(df_manga, index):

  if df_manga['No. NVSS matches(3m)'][index] != 4:
    return False

  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]
  c_manga = SkyCoord(ra_manga, dec_manga,frame = 'icrs',unit = 'degree')

  ra1 = df_manga['RA NVSS(3m)'][index][0]
  dec1 = df_manga['DEC NVSS(3m)'][index][0]

  ra2 = df_manga['RA NVSS(3m)'][index][1]
  dec2 = df_manga['DEC NVSS(3m)'][index][1]

  ra3 = df_manga['RA NVSS(3m)'][index][2]
  dec3 = df_manga['DEC NVSS(3m)'][index][2]

  ra4 = df_manga['RA NVSS(3m)'][index][3]
  dec4 = df_manga['DEC NVSS(3m)'][index][3]

  f1 = df_manga['Flux NVSS(3m)'][index][0]
  f2 = df_manga['Flux NVSS(3m)'][index][1]
  f3 = df_manga['Flux NVSS(3m)'][index][2]
  f4 = df_manga['Flux NVSS(3m)'][index][3]

  [ra_fwmp, dec_fwmp] = fwmp_4(ra1,dec1,ra2,dec2,ra3,dec3,ra4,dec4,f1,f2,f3,f4)

  c_fwmp = SkyCoord(ra_fwmp,dec_fwmp,frame = 'icrs', unit = 'degree')

  sep_fwmp = c_manga.separation(c_fwmp).value*3600

  if sep_fwmp < 30:
    return True
  
  else:
    return False


def is_nvss_quad(df_manga, index):

  if is_candidate_quad(df_manga, index) != True:
    return 0
  
  else:
    return "Needs VI 4(4)"
  

def is_candidate_triple_4matches(df_manga, index, ra1,dec1,ra2,dec2,ra3,dec3,f1,f2,f3):

  if df_manga['No. NVSS matches(3m)'][index] !=3:
    return False
        
  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]

  c1 = SkyCoord(ra1, dec1, frame='icrs', unit = 'degree')
  c2 = SkyCoord(ra2, dec2, frame='icrs', unit = 'degree')
  c3 = SkyCoord(ra3, dec3, frame='icrs', unit = 'degree')

  c_manga = SkyCoord(ra_manga, dec_manga, frame='icrs', unit = 'degree')

  sep1 = c_manga.separation(c1).value*3600
  sep2 = c_manga.separation(c2).value*3600
  sep3 = c_manga.separation(c3).value*3600  

 
  
  if (sep1 < 120) and (sep2 < 120) and (sep3 < 120):
    

    [ra_fwmp, dec_fwmp] = fwmp_3(ra1,dec1,ra2,dec2,ra3,dec3,f1,f2,f3)
    c_fwmp = SkyCoord(ra_fwmp, dec_fwmp, frame = 'icrs', unit = 'degree')
    sep_fwmp = c_manga.separation(c_fwmp).value*3600
    
    
    if sep_fwmp < 15:
      
      return True
    
    
    seps_array = np.array([sep1, sep2, sep3])
    inds_sort = seps_array.argsort()
    seps_array = seps_array[inds_sort]

    ras = np.array([ra1,ra2,ra3])[inds_sort]
    decs = np.array([dec1, dec2, dec3])[inds_sort]
    fluxs = np.array([f1, f2, f3])[inds_sort]

    ra_near = ras[0]
    ra_mid = ras[1]
    ra_far = ras[2]

    dec_near = decs[0]
    dec_mid = decs[1]
    dec_far = decs[2]

    f_near = fluxs[0]
    f_mid = fluxs[1]
    f_far = fluxs[2]

    [ra_fwmp_far2, dec_fwmp_far2] = fwmp_2(ra_mid,dec_mid,ra_far,dec_far,f_mid,f_far)
    c_fwmp_far2 = SkyCoord(ra_fwmp_far2, dec_fwmp_far2,frame = 'icrs',unit = 'degree')
    sep_fwmp_far2 = c_manga.separation(c_fwmp_far2).value*3600


    if sep_fwmp_far2 < 15:
      return True
    
    sep_near = seps_array[0]
    sep_mid = seps_array[1]
    sep_far = seps_array[2]

    if sep_near < 15:
      
      if get_agn_angle(ra_mid,dec_mid, ra_far,dec_far,ra_near,dec_near) > 135:
      
        if sep_mid < 90 and sep_far < 90:
      
          return True
      
      else:
        return False
    else:
      return False
    

  else:
    return False  
  
  
def is_nvss_triple_4matches(df_manga,index,ra1,dec1,ra2,dec2,ra3,dec3,f1,f2,f3):

  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]
  c_manga = SkyCoord(ra_manga,dec_manga,frame='icrs',unit = 'degree')

  if df_manga['No. NVSS matches(3m)'][index] !=3:
    return 0
  
  if is_candidate_triple_4matches(df_manga,index,ra1,dec1,ra2,dec2,ra3,dec3,f1,f2,f3) == False:
    return 0

  if df_manga['3 arcsec FIRST match'][index] == 1:
    return 4
  
  if (df_manga['No. FIRST matches(30s)'][index] <= 3) and (df_manga['No. FIRST matches(30s)'][index] > 0):
    
    seps = []

    for i in range(len(df_manga['RA FIRST(30s)'][index])):

      ra_i = df_manga['RA FIRST(30s)'][index][i]
      dec_i = df_manga['DEC FIRST(30s)'][index][i]

      c_i = SkyCoord(ra_i, dec_i, frame = 'icrs', unit = 'degree')

      sep = c_manga.separation(c_i).value*3600
      seps.append(sep)
      
    seps = np.array(seps)

    if (seps > 15).all():

      if (df_manga['Flux FIRST(30s)'][index].sum() > 0.5*df_manga['Flux NVSS(30s)'][index].sum()):
        return 0
      
      else:
        return 0
        
    else:
        return "Needs VI 4(3)"

  else:
    return "Needs VI 4(3)"
  

def indices_triple_4matches(df_manga, index):

  if is_candidate_quad(df_manga,index) != True:
    return 0

  if is_nvss_quad(df_manga,index) != False:
    return 0
  
  #1st and 2nd and 3rd galaxies out of the four matches

  ra_manga = df_manga['RA']
  dec_manga = df_manga['DEC']

  ra1 = df_manga['RA NVSS(3m)'][index][0]
  dec1 = df_manga['DEC NVSS(3m)'][index][0]

  ra2 = df_manga['RA NVSS(3m)'][index][1]
  dec2 = df_manga['DEC NVSS(3m)'][index][1]

  ra3 = df_manga['RA NVSS(3m)'][index][2]
  dec3 = df_manga['DEC NVSS(3m)'][index][2]

  f1 = df_manga['Flux NVSS(3m)'][index][0]
  f2 = df_manga['Flux NVSS(3m)'][index][1]
  f3 = df_manga['Flux NVSS(3m)'][index][2]   

  if is_nvss_triple_4matches(df_manga,index,ra1,dec1,ra2,dec2,ra3,dec3,f1,f2,f3) == True:
    return [0,1,2]
  
  #2nd and 3rd and 4th galaxies out of the four matches
  ra1 = df_manga['RA NVSS(3m)'][index][1]
  dec1 = df_manga['DEC NVSS(3m)'][index][1]

  ra2 = df_manga['RA NVSS(3m)'][index][2]
  dec2 = df_manga['DEC NVSS(3m)'][index][2]

  ra3 = df_manga['RA NVSS(3m)'][index][3]
  dec3 = df_manga['DEC NVSS(3m)'][index][3]

  f1 = df_manga['Flux NVSS(3m)'][index][1]
  f2 = df_manga['Flux NVSS(3m)'][index][2]
  f3 = df_manga['Flux NVSS(3m)'][index][3]

  if is_nvss_triple_4matches(df_manga,index,ra1,dec1,ra2,dec2,ra3,dec3,f1,f2,f3) == True:
    return [1,2,3]
  
  #1st, 2nd and 4th galaxies out of the three matches
  ra1 = df_manga['RA NVSS(3m)'][index][0]
  dec1 = df_manga['DEC NVSS(3m)'][index][0]

  ra2 = df_manga['RA NVSS(3m)'][index][1]
  dec2 = df_manga['DEC NVSS(3m)'][index][1]

  ra3 = df_manga['RA NVSS(3m)'][index][3]
  dec3 = df_manga['DEC NVSS(3m)'][index][3]

  f1 = df_manga['Flux NVSS(3m)'][index][0]
  f2 = df_manga['Flux NVSS(3m)'][index][1]
  f3 = df_manga['Flux NVSS(3m)'][index][3]    

  if is_nvss_triple_4matches(df_manga,index,ra1,dec1,ra2,dec2,ra3,dec3,f1,f2,f3) == True:
    return [0,1,3]

  ra1 = df_manga['RA NVSS(3m)'][index][0]
  dec1 = df_manga['DEC NVSS(3m)'][index][0]

  ra2 = df_manga['RA NVSS(3m)'][index][2]
  dec2 = df_manga['DEC NVSS(3m)'][index][2]

  ra3 = df_manga['RA NVSS(3m)'][index][3]
  dec3 = df_manga['DEC NVSS(3m)'][index][3]

  f1 = df_manga['Flux NVSS(3m)'][index][0]
  f2 = df_manga['Flux NVSS(3m)'][index][2]
  f3 = df_manga['Flux NVSS(3m)'][index][3]    

  if is_nvss_triple_4matches(df_manga,index,ra1,dec1,ra2,dec2,ra3,dec3,f1,f2,f3) == True:
    return [0,2,3]
  
  else:
    return 0

############################################

#More than 4 NVSS matches

def is_candidate_multiple(df_manga,index):

  if df_manga['No. NVSS matches(3m)'][index] > 4:
    return True
  
  else:
    return 0
  

def is_nvss_multiple(df_manga,index):

  if is_candidate_multiple(df_manga, index) != True:

    return 0
  
  else:

    return "Needs VI 4(4+)"

##########################################################################

#NVSS Single

def is_candidate_single_two_matches(df_matched,index):
    
  if type(is_candidate_double(df_matched,index)) == list:
    return True
  
  else:
    return False
  
  
def is_candidate_single(df_manga,index):

  if df_manga['No. NVSS matches(3m)'][index] != 1:
    return False
  
  ra1 = df_manga['RA NVSS(3m)'][index]
  dec1 = df_manga['DEC NVSS(3m)'][index]

  c_1 = SkyCoord(ra1,dec1,frame = 'icrs', unit = 'degree')

  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]

  c_manga = SkyCoord(ra_manga,dec_manga,frame = 'icrs', unit = 'degree')

  sep1 = c_manga.separation(c_1).value*3600

  if sep1 < 30:
    return True
  
  else:
    return False
  
  
def is_nvss_single_0F(df_manga,index):

  if is_candidate_single(df_manga, index) != True:
    return 0
  
  if df_manga['No. FIRST matches(30s)'][index] == 0:
    ra_manga = df_manga['RA'][index]
    dec_manga = df_manga['DEC'][index]
    c_manga = SkyCoord(ra_manga, dec_manga, frame = 'icrs', unit = 'degree')

    ra_nvss = df_manga['RA NVSS(3m)'][index][0]
    dec_nvss = df_manga['DEC NVSS(3m)'][index][0]
      
    c_1 = SkyCoord(ra_nvss,dec_nvss, frame = 'icrs', unit = 'degree')
    sep1 = c_manga.separation(c_1).value*3600

    if sep1 < 10:
      return 3
    
    else:
      return 0

  else:
    return 0


def get_end_of_jet(ra_first,dec_first,pa,maj):

  pa = pa*np.pi/180
    
  dec2 = (dec_first*3600 + (maj/2)*np.cos(pa))/3600
  
  ra2 = (ra_first*3600 + (maj/2)*np.sin(pa))/3600
    
  return [ra2, dec2]


def is_nvss_single_1F(df_manga,index):

  if is_candidate_single(df_manga, index) != True:
    return 0
  
  if df_manga['No. FIRST matches(30s)'][index] != 1:
    return 0
  
  if df_manga['3 arcsec FIRST match'][index] == 1:
    return 1
  
  if df_manga['No. FIRST matches(30s)'][index] == 1:

    ra_manga = df_manga['RA'][index]
    dec_manga = df_manga['DEC'][index]

    ra_f1 = df_manga['RA FIRST(30s)'][index][0]
    dec_f1 = df_manga['DEC FIRST(30s)'][index][0]

    c_1 = SkyCoord(ra_f1, dec_f1, frame = 'icrs', unit = 'degree')
    c_manga = SkyCoord(ra_manga, dec_manga, frame = 'icrs', unit = 'degree')

    sep1 = c_manga.separation(c_1).value*3600

    if sep1 < 10:

      pa = df_manga['PA FIRST(30s)'][index][0]
      maj = df_manga['Maj FIRST(30s)'][index][0]

      [ra_jet, dec_jet] = get_end_of_jet(ra_f1, dec_f1, pa, maj)

      
      jet_angle = get_agn_angle(ra_manga,dec_manga,ra_jet,dec_jet,ra_f1,dec_f1)
      jet_angle = jet_angle*np.pi/180

      if np.cos(jet_angle) < 0:
        jet_angle = np.pi - jet_angle
     
      if sep1 < 0.75*maj*np.cos(jet_angle):

        if (0 < jet_angle < 30*np.pi/180) or (150*np.pi/180 < jet_angle < np.pi):

          return 1
    
        else:

          return 0
        
      else:
        return 0
    
    else:
      return 0
  
  else:
    return 0


def is_nvss_single_2F(df_manga, index):

  if is_candidate_single(df_manga,index) != True:
    return 0
  

  if df_manga['No. FIRST matches(30s)'][index] == 2:
    
    ra_f1 = df_manga['RA FIRST(30s)'][index][0]
    dec_f1 = df_manga['DEC FIRST(30s)'][index][0]

    ra_f2 = df_manga['RA FIRST(30s)'][index][1]
    dec_f2 = df_manga['DEC FIRST(30s)'][index][1]

    c_f1 = SkyCoord(ra_f1,dec_f1,frame = 'icrs',unit = 'degree')
    c_f2 = SkyCoord(ra_f2,dec_f2,frame = 'icrs',unit = 'degree')  

    ra_manga = df_manga['RA'][index]
    dec_manga = df_manga['DEC'][index]

    c_manga = SkyCoord(ra_manga,dec_manga,frame = 'icrs',unit = 'degree')

    sep_f1 = c_manga.separation(c_f1).value*3600
    sep_f2 = c_manga.separation(c_f2).value*3600

    seps = np.array([sep_f1,sep_f2])

    sep_near = seps.min()
    sep_far = seps.max()

    if sep_near < 3:
      return 2
    
    flux_1 = df_manga['Flux FIRST(30s)'][index][0]
    flux_2 = df_manga['Flux FIRST(30s)'][index][1]

    fluxs = np.array([flux_1,flux_2])
    sort_inds = np.argsort(fluxs)

    fluxs = fluxs[sort_inds]

    flux_low = fluxs[0]
    flux_high = fluxs[1]

    maj1 = df_manga['Maj FIRST(30s)'][index][0]
    maj2 = df_manga['Maj FIRST(30s)'][index][1]

    majs = np.array([maj1,maj2])[sort_inds]

    maj_low = majs[0]
    maj_high = majs[1]

    [ra_fwmp, dec_fwmp] = fwmp_2(ra_f1,dec_f1,ra_f2,dec_f2,flux_1,flux_2)
    c_fwmp = SkyCoord(ra_fwmp,dec_fwmp,frame = 'icrs', unit = 'degree')
    sep_fwmp = c_manga.separation(c_fwmp).value*3600

    char_quant = (flux_high/flux_low)*(maj_high/maj_low)*(sep_fwmp)

    if char_quant < 5:
      return 2
    
    else:
      return 0
  
  else:
    return 0
  

def is_nvss_single_2F_3matches(df_manga,index,ra_f1,dec_f1,ra_f2,dec_f2,flux_1,flux_2,maj1,maj2):

  if is_candidate_single(df_manga,index) != True:
    return 0
  
  if df_manga['No. FIRST matches(30s)'][index] == 2:

    c_f1 = SkyCoord(ra_f1,dec_f1,frame = 'icrs',unit = 'degree')
    c_f2 = SkyCoord(ra_f2,dec_f2,frame = 'icrs',unit = 'degree')  

    ra_manga = df_manga['RA'][index]
    dec_manga = df_manga['DEC'][index]

    c_manga = SkyCoord(ra_manga,dec_manga,frame = 'icrs',unit = 'degree')

    sep_f1 = c_manga.separation(c_f1).value*3600
    sep_f2 = c_manga.separation(c_f2).value*3600

    seps = np.array([sep_f1,sep_f2])

    sep_near = seps.min()
    sep_far = seps.max()

    if sep_near < 3:
      return 2
    
    flux_1 = df_manga['Flux FIRST(30s)'][index][0]
    flux_2 = df_manga['Flux FIRST(30s)'][index][1]
    fluxs = np.array([flux_1,flux_2])
    sort_inds = np.argsort(fluxs)

    fluxs = fluxs[sort_inds]

    flux_low = fluxs[0]
    flux_high = fluxs[1]

    maj1 = df_manga['Maj FIRST(30s)'][index][0]
    maj2 = df_manga['Maj FIRST(30s)'][index][1]
    majs = np.array([maj1,maj2])[sort_inds]

    maj_low = majs[0]
    maj_high = majs[1]

    maj1 = df_manga['Maj FIRST(30s)'][index][0]
    maj2 = df_manga['Maj FIRST(30s)'][index][1]

    [ra_fwmp, dec_fwmp] = fwmp_2(ra_f1,dec_f1,ra_f2,dec_f2,flux_1,flux_2)
    c_fwmp = SkyCoord(ra_fwmp,dec_fwmp,frame = 'icrs', unit = 'degree')
    sep_fwmp = c_manga.separation(c_fwmp).value*3600

    char_quant = (flux_low/flux_high)*(maj_low/maj_high)*(sep_fwmp)

    if char_quant < 5:
      return 2
    
    else:
      return 0
  
  else:
    return 0
  

def is_nvss_single_3F(df_manga,index):

  if is_candidate_single(df_manga, index) != True:
    return 0

  if df_manga['No. FIRST matches(30s)'][index] != 3:
    return 0
  
  ra_f1 = df_manga['RA FIRST(30s)'][index][0]
  dec_f1 = df_manga['DEC FIRST(30s)'][index][0]

  ra_f2 = df_manga['RA FIRST(30s)'][index][1]
  dec_f2 = df_manga['DEC FIRST(30s)'][index][1]

  ra_f3 = df_manga['RA FIRST(30s)'][index][2]
  dec_f3 = df_manga['DEC FIRST(30s)'][index][2]

  [flux1, flux2,flux3] = df_manga['Flux FIRST(30s)'][index]


  c_f1 = SkyCoord(ra_f1,dec_f1,frame = 'icrs',unit = 'degree')
  c_f2 = SkyCoord(ra_f2,dec_f2,frame = 'icrs',unit = 'degree')
  c_f3 = SkyCoord(ra_f3,dec_f3,frame = 'icrs',unit = 'degree')

  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]

  c_manga = SkyCoord(ra_manga, dec_manga, frame = 'icrs', unit = 'degree')

  sep1 = c_manga.separation(c_f1).value*3600
  sep2 = c_manga.separation(c_f2).value*3600
  sep3 = c_manga.separation(c_f3).value*3600

  seps = np.array([sep1, sep2, sep3])
  inds_sort = np.argsort(seps)
  seps = seps[inds_sort]

  sep_near = seps[0]
  sep_mid = seps[1]
  sep_far = seps[2]

  ras = df_manga['RA FIRST(30s)'][index]
  decs = df_manga['DEC FIRST(30s)'][index]

  ras = np.array(ras)[inds_sort]
  decs = np.array(decs)[inds_sort]

  [ra_near,ra_mid,ra_far] = ras
  [dec_near, dec_mid, dec_far] = decs

  if sep_near < 3:
    return 2
  
  [ra_fwmp, dec_fwmp] = fwmp_3(ra_f1,dec_f1,ra_f2,dec_f2,ra_f3,dec_f3,flux1,flux2,flux3)
  c_fwmp = SkyCoord(ra_fwmp,dec_fwmp,frame = 'icrs', unit = 'degree')
  
  sep_fwmp = c_manga.separation(c_fwmp).value*3600
  
  if sep_fwmp < 3:

    angle_agn = get_agn_angle(ra_mid,dec_mid,ra_far,dec_far,ra_near,dec_near)

    if angle_agn > 135:

      return 2
    
    else:

      return 0
  
  ra_f1 = df_manga['RA FIRST(30s)'][index][0]
  dec_f1 = df_manga['DEC FIRST(30s)'][index][0]

  ra_f2 = df_manga['RA FIRST(30s)'][index][1]
  dec_f2 = df_manga['DEC FIRST(30s)'][index][1]

  flux1 = df_manga['Flux FIRST(30s)'][index][0]
  flux2 = df_manga['Flux FIRST(30s)'][index][1]

  maj1 = df_manga['Maj FIRST(30s)'][index][0]
  maj2 = df_manga['Maj FIRST(30s)'][index][1]

  if is_nvss_single_2F_3matches(df_manga,index,ra_f1,
  dec_f1,ra_f2,dec_f2,flux1,flux2,maj1,maj2) == 2:
    
    return 2
  
  ra_f1 = df_manga['RA FIRST(30s)'][index][0]
  dec_f1 = df_manga['DEC FIRST(30s)'][index][0]

  ra_f2 = df_manga['RA FIRST(30s)'][index][2]
  dec_f2 = df_manga['DEC FIRST(30s)'][index][2]

  flux1 = df_manga['Flux FIRST(30s)'][index][0]
  flux2 = df_manga['Flux FIRST(30s)'][index][2]

  maj1 = df_manga['Maj FIRST(30s)'][index][0]
  maj2 = df_manga['Maj FIRST(30s)'][index][2]

  if is_nvss_single_2F_3matches(df_manga,index,ra_f1,
  dec_f1,ra_f2,dec_f2,flux1,flux2,maj1,maj2) == True:
    
    return 2
  
  ra_f1 = df_manga['RA FIRST(30s)'][index][1]
  dec_f1 = df_manga['DEC FIRST(30s)'][index][1]

  ra_f2 = df_manga['RA FIRST(30s)'][index][2]
  dec_f2 = df_manga['DEC FIRST(30s)'][index][2]

  flux1 = df_manga['Flux FIRST(30s)'][index][1]
  flux2 = df_manga['Flux FIRST(30s)'][index][2]

  maj1 = df_manga['Maj FIRST(30s)'][index][1]
  maj2 = df_manga['Maj FIRST(30s)'][index][2]

  if is_nvss_single_2F_3matches(df_manga,index,ra_f1,
  dec_f1,ra_f2,dec_f2,flux1,flux2,maj1,maj2) == True:
    
    return 2
  
  else:

    return 0
  

def is_nvss_single_4F(df_manga, index):

    if is_candidate_single(df_manga,index) != True:
      return 0

    if df_manga['No. FIRST matches(30s)'][index] < 4:
      return 0
    
    num_F_matches = df_manga['No. FIRST matches(30s)'][index]

    ras = df_manga['RA FIRST(30s)'][index]
    decs = df_manga['DEC FIRST(30s)'][index]

    fluxs = df_manga['Flux FIRST(30s)'][index]
    majs = df_manga['Maj FIRST(30s)'][index]

    ra_manga = df_manga['RA'][index]
    dec_manga = df_manga['DEC'][index]

    c_manga = SkyCoord(ra_manga, dec_manga, frame = 'icrs', unit = 'degree')
    c_mats = SkyCoord(ras, decs, frame = 'icrs', unit = 'degree')

    seps = c_manga.separation(c_mats).value*3600

    ind_sort = np.argsort(seps)

    seps = seps[ind_sort]

    ras = ras[ind_sort]
    decs = decs[ind_sort]

    fluxs = fluxs[ind_sort]
    majs = majs[ind_sort]

    ra_f1,ra_f2,ra_f3 = ras[0:3]
    dec_f1,dec_f2,dec_f3 = decs[0:3]
    f1,f2,f3 = fluxs[0:3]
    maj_1,maj_2,maj_3 = majs[0:3]

    if is_nvss_single_3F_4matches(df_manga,index,ra_f1,dec_f1,ra_f2,
    dec_f2,ra_f3,dec_f3,f1,f2,f3,maj_1,maj_2,maj_3) == 2:
      
      return 2
    
    else:
      
      return "Needs VI 2(4+)"



def is_nvss_single_3F_4matches(df_manga,index,ra_f1,dec_f1, ra_f2, dec_f2, ra_f3,dec_f3,f1,f2,f3,
maj_1, maj_2, maj_3):

  if is_candidate_single(df_manga, index) != True:
    return 0

  if df_manga['No. FIRST matches(30s)'][index] != 3:
    return 0
  
  c_f1 = SkyCoord(ra_f1,dec_f1,frame = 'icrs',unit = 'degree')
  c_f2 = SkyCoord(ra_f2,dec_f2,frame = 'icrs',unit = 'degree')
  c_f3 = SkyCoord(ra_f3,dec_f3,frame = 'icrs',unit = 'degree')

  ra_manga = df_manga['RA'][index]
  dec_manga = df_manga['DEC'][index]

  c_manga = SkyCoord(ra_manga, dec_manga, frame = 'icrs', unit = 'degree')

  sep1 = c_manga.separation(c_f1).value*3600
  sep2 = c_manga.separation(c_f2).value*3600
  sep3 = c_manga.separation(c_f3).value*3600

  seps = np.array([sep1, sep2, sep3])
  inds_sort = np.argsort(seps)
  seps = seps[inds_sort]

  sep_near = seps[0]
  sep_mid = seps[1]
  sep_far = seps[2]

  ras = df_manga['RA FIRST(30s)'][index]
  decs = df_manga['DEC FIRST(30s)'][index]

  ras = np.array(ras)[inds_sort]
  decs = np.array(decs)[inds_sort]

  [ra_near,ra_mid,ra_far] = ras
  [dec_near, dec_mid, dec_far] = decs

  if sep_near < 3:
    return 2
  
  [ra_fwmp, dec_fwmp] = fwmp_3(ra_f1,dec_f1,ra_f2,dec_f2,ra_f3,dec_f3,f1,f2,f3)
  c_fwmp = SkyCoord(ra_fwmp,dec_fwmp,frame = 'icrs', unit = 'degree')
  
  sep_fwmp = c_manga.separation(c_fwmp).value*3600
  
  if sep_fwmp < 3:

    angle_agn = get_agn_angle(ra_mid,dec_mid,ra_far,dec_far,ra_near,dec_near)

    if angle_agn > 135:

      return 2
    
    else:

      return 0
  
  ra1 = ra_f1
  dec1 = dec_f1

  ra2 = ra_f2
  dec2 = dec_f2

  flux1 = f1
  flux2 = f2

  maj1 = maj_1
  maj2 = maj_2

  if is_nvss_single_2F_3matches(df_manga,index,ra1,
  dec1,ra2,dec2,flux1,flux2,maj1,maj2) == True:
    
    return 2
  
  ra1 = ra_f2
  dec1 = dec_f2

  ra2 = ra_f3
  dec2 = dec_f3

  flux1 = f2
  flux2 = f3

  maj1 = maj_2
  maj2 = maj_3

  if is_nvss_single_2F_3matches(df_manga,index,ra1,
  dec1,ra2,dec2,flux1,flux2,maj1,maj2) == True:
    
    return 2
  
  ra1 = ra_f1
  dec1 = dec_f1

  ra2 = ra_f3
  dec2 = dec_f3

  flux1 = f1
  flux2 = f3

  maj1 = maj_1
  maj2 = maj_3

  if is_nvss_single_2F_3matches(df_manga,index,ra1,
  dec1,ra2,dec2,flux1,flux2,maj1,maj2) == True:
    
    return 2
  
  else:

    return 0
  
########################################################

def get_radio_class(df_manga, index):

  if is_nvss_single_0F(df_manga,index) == 3:
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return 3
    
    else:
      return 0
  
  if is_nvss_single_1F(df_manga,index) == 1:
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:    
      return 1
    
    else:
      return 0
  
  if is_nvss_single_2F(df_manga,index) == 2:
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return 2
    
    else:
      return 0
  
  if is_nvss_single_3F(df_manga,index) == 2:
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return 2

    else:
      return 0
  
  if is_nvss_single_4F(df_manga,index) == 2:
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return 2
    
    else:
      return 0
  
  if is_nvss_single_4F(df_manga, index) == "Needs VI 2(4+)":
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return "Needs VI 2(4+)"
    
    else:
      return 0
  
  if is_nvss_double(df_manga,index) == 4:
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return 4
    
    else:
      return 0
  
  if is_nvss_double(df_manga, index) == "Needs VI 4(2)":

    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return "Needs VI 4(2)"
    
    else:
      return 0
    
  if is_nvss_triple(df_manga,index) == 4:

    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return 4
    
    else:
      return 0
  

  if is_nvss_triple(df_manga, index) == "Needs VI 4(3)":
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return "Needs VI 4(3)"
    
    else:
      return 0

  
  if is_nvss_triple(df_manga, index) == "Needs VI 4(2)":
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return "Needs VI 4(2)"
    
    else:
      return 0

  
  if is_nvss_quad(df_manga,index) == 4:
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return 4
    
    else:
      return 0
  
  if is_nvss_quad(df_manga,index) == "Needs VI 4(4)":
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return "Needs VI 4(4)"
    
    else:
      return 0
  
  if is_nvss_quad(df_manga,index) == "Needs VI 4(3)":
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return "Needs VI 4(3)"
    
    else:
      return 0

  
  if is_nvss_multiple(df_manga,index) == 4:
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return 4
    
    else:
      return 0
  
  if is_nvss_multiple(df_manga, index) == "Needs VI 4(4+)":

    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    if flux_nvss > 2.5:
      return "Needs VI 4(4+)"
    
    else:
      return 0

  
  if is_vi_source(df_manga,index) == True:
    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    n_nvss = df_manga['No. NVSS matches(3m)'][index]
    n_first = df_manga['No. FIRST matches(30s)'][index]


    if flux_nvss > 2.5:
      if n_nvss == 1 and n_first == 1:
        return 1
    
      elif n_nvss == 1 and n_first > 1:
        return 2
    
      elif n_nvss == 1 and n_first == 0:
        return 3
    
      elif n_nvss > 1:
        return 4
      
      else:
        return 0
      
    else:
      return 0
  
  else:
    return 0
  
#######################################

def load_drpmatched_df():

  df_drpall = load_drpall_df()

  for i in range(10261):
    delete_unrelated_flux_2_1(df_drpall,i)

  for i in range(10261):
    delete_unrelated_flux_3_1(df_drpall,i)
  
  for i in range(10261):
    delete_unrelated_nvss_flux_vi(df_drpall,i)

  radio_class = []

  for i in range(10261):
    if type(get_radio_class(df_drpall,i)) == str:

      if is_vi_source(df_drpall, i) == True:
        radio_class.append(classify_vi_source(df_drpall,i))
      
      else:
        radio_class.append(0)
    
    else:
      radio_class.append(get_radio_class(df_drpall,i))
  
  radio_class = np.array(radio_class, dtype = object)
  df_drpall['Radio Class'] = radio_class

  flux = []

  for i in range(10261):
    flux.append(get_nvss_flux(df_drpall,i))

  flux = np.array(flux)

  df_drpall['Flux'] = flux


  
  return df_drpall

    
def get_nvss_flux(df_matched, index):
  
  if get_radio_class(df_matched,index) != 0:

    if type(get_radio_class(df_matched,index)) != str:

      flux_array = np.array(df_matched['Flux NVSS(3m)'][index])

      flux = flux_array.sum()
      
      if flux > 2.5:
        return flux

      else:
        return 0


    elif type(get_radio_class(df_matched,index)) == str:

      if is_vi_source(df_matched,index) == True:
        flux_array = np.array(df_matched['Flux NVSS(3m)'][index])
        flux = flux_array.sum()
      
        if flux > 2.5:
          return flux

        else:
          return 0
      
      else:
        return 0
    
    else:
      return 0

  
  elif get_radio_class(df_matched,index) == 0:

    if is_vi_source(df_matched,index) == True:
      flux_array = np.array(df_matched['Flux NVSS(3m)'][index])
      flux = flux_array.sum()
      
      if flux > 2.5:
        return flux

      else:
        return 0
      
    else:
      return 0
      
  else:
    return 0


###########################################



############################################

def get_nvss_2_seps(df_matched):

  ind_2_matches = np.where(df_matched['No. NVSS matches(3m)'] == 2)[0]

  ras1 = []
  decs1 = []
  ras2 = []
  decs2 = []

  for i in df_matched['RA NVSS(3m)'][ind_2_matches]:

    ras1.append(i[0])
    ras2.append(i[1])

  for j in df_matched['DEC NVSS(3m)'][ind_2_matches]:

    decs1.append(j[0])
    decs2.append(j[1])

  ras1 = np.array(ras1)
  decs1 = np.array(decs1)

  ras2 = np.array(ras2)
  decs2 = np.array(decs2)

  ras_manga = df_matched['RA'][ind_2_matches]
  decs_manga = df_matched['DEC'][ind_2_matches]

  c_1 = SkyCoord(ras1,decs1,frame = 'icrs', unit = 'degree')
  c_2 = SkyCoord(ras2,decs2,frame = 'icrs', unit = 'degree')
  c_manga = SkyCoord(ras_manga,decs_manga,frame = 'icrs', unit = 'degree')

  seps1 = c_manga.separation(c_1).value*3600
  seps2 = c_manga.separation(c_2).value*3600

  seps_near = []
  seps_far = []

  for i in range(len(seps1)):

    if seps1[i] < seps2[i]:

      seps_near.append(seps1[i])
      seps_far.append(seps2[i])
        
    else:

      seps_near.append(seps2[i])
      seps_far.append(seps1[i])

  seps_near = np.array(seps_near)
  seps_far = np.array(seps_far)

  ind_cands = []
  for i in range(len(ind_2_matches)):
    if is_candidate_double(df_matched.iloc[ind_2_matches],ind_2_matches[i]) == True:
      ind_cands.append(i)
  
  ind_cands = np.array(ind_cands)
  ind_sources = np.where(df_matched['Radio Class'][ind_2_matches] == 4)
  ind_vi = np.where(df_matched['Radio Class'][ind_2_matches] == "Needs VI 4(2)")

  return [seps_near, seps_far, ind_sources, ind_vi, ind_cands]


def nvss_double_candidate_cut(near_offset):

  if near_offset < 15:

    far_offset = near_offset + 20

  elif near_offset > 15:

    far_offset = 90
  
  return far_offset

v_nvss_double_candidate_cut = np.vectorize(nvss_double_candidate_cut)

def get_nvss_3_seps(df_matched):

  ind_3_matches = np.where(df_matched['No. NVSS matches(3m)'] == 3)[0]

  ras1 = []
  decs1 = []
  ras2 = []
  decs2 = []
  ras3 = []
  decs3 = []

  for i in df_matched['RA NVSS(3m)'][ind_3_matches]:

    ras1.append(i[0])
    ras2.append(i[1])
    ras3.append(i[2])

  for j in df_matched['DEC NVSS(3m)'][ind_3_matches]:

    decs1.append(j[0])
    decs2.append(j[1])
    decs3.append(j[2])

  ras1 = np.array(ras1)
  decs1 = np.array(decs1)

  ras2 = np.array(ras2)
  decs2 = np.array(decs2)

  ras3 = np.array(ras3)
  decs3 = np.array(decs3)

  ras_manga = df_matched['RA'][ind_3_matches]
  decs_manga = df_matched['DEC'][ind_3_matches]

  c_1 = SkyCoord(ras1,decs1,frame = 'icrs', unit = 'degree')
  c_2 = SkyCoord(ras2,decs2,frame = 'icrs', unit = 'degree')
  c_3 = SkyCoord(ras3,decs3,frame = 'icrs', unit = 'degree')
  c_manga = SkyCoord(ras_manga,decs_manga,frame = 'icrs', unit = 'degree')

  seps1 = c_manga.separation(c_1).value*3600
  seps2 = c_manga.separation(c_2).value*3600
  seps3 = c_manga.separation(c_3).value*3600

  
  seps_mid = []
  seps_far = []

  for i in range(len(seps2)):

    temp_array = np.array([seps1[i],seps2[i],seps3[i]])
    temp_array = np.sort(temp_array)
    seps_mid.append(temp_array[1])
    seps_far.append(temp_array[-1])

  seps_mid = np.array(seps_mid)
  seps_far = np.array(seps_far)

  ind_cands = []
  for i in range(len(ind_3_matches)):
    if is_candidate_triple(df_matched.iloc[ind_3_matches],ind_3_matches[i]) == True:
      ind_cands.append(i)
  
  ind_cands = np.array(ind_cands)
  ind_sources = np.where(df_matched['Radio Class'][ind_3_matches] == 4)[0]
  ind_vi = np.where(df_matched['Radio Class'][ind_3_matches] == "Needs VI 4(3)")[0]
  

  return [seps_mid, seps_far, ind_sources, ind_vi, ind_cands]


def get_nearest_nvss_sep(df_matched, index):
    
  if df_matched['No. NVSS matches(3m)'][index] != 0:

    if df_matched['Radio Class'][index] != 4:

      ras = df_matched['RA NVSS(3m)'][index]
      decs = df_matched['DEC NVSS(3m)'][index]

      c = SkyCoord(ras, decs, frame = 'icrs', unit = 'degree')
      c_manga = SkyCoord(df_matched['RA'][index], df_matched['DEC'][index], frame = 'icrs', unit = 'degree')

      seps = c_manga.separation(c).value*3600

      return seps.min()


def nvss_single1_cut(proj_maj):
    
  if proj_maj < 4:

    near_sep = 3

  if proj_maj > 4.0:

    near_sep = 0.75*proj_maj

  
  if proj_maj > 4*10/3:
    near_sep = 10
  
  return float(near_sep)
    

v_nvss_single1_cut = np.vectorize(nvss_single1_cut)


def get_source1_seps_and_pma(df_matched):

  inds_1match = np.where(df_matched['No. FIRST matches(30s)'] == 1)[0]

  seps = []
  proj = []
  ind_source = []

  for i in range(len(inds_1match)):
  
    ra_first = df_matched['RA FIRST(30s)'][inds_1match[i]]
    dec_first = df_matched['DEC FIRST(30s)'][inds_1match[i]]

    ra_manga = df_matched['RA'][inds_1match[i]]
    dec_manga = df_matched['DEC'][inds_1match[i]]

    c_first = SkyCoord(ra_first, dec_first, frame = 'icrs', unit = 'degree')
    c_manga = SkyCoord(ra_manga, dec_manga, frame = 'icrs', unit = 'degree')
    seps.append(c_manga.separation(c_first).value*3600)

    pa = df_matched['PA FIRST(30s)'][inds_1match[i]]
    maj = df_matched['Maj FIRST(30s)'][inds_1match[i]]
    [ra_jet, dec_jet] = get_end_of_jet(ra_first, dec_first,pa,maj)

    jet_angle = get_agn_angle(ra_manga,dec_manga,ra_jet,dec_jet,ra_first,dec_first)

    jet_angle = jet_angle*np.pi/180

    if np.cos(jet_angle) < 0:
      jet_angle = np.pi - jet_angle

    proj_maj = maj*np.cos(jet_angle)
    proj.append(proj_maj)

    if is_nvss_single_1F(df_matched,inds_1match[i]) == 1:
      ind_source.append(i)

  seps = np.array(seps)
  proj = np.array(proj)
  ind_source = np.array(ind_source)

  return [seps, proj, ind_source]



def get_first_2_seps(df_matched):

  ind_2_matches = np.where(df_matched['No. FIRST matches(30s)'] == 2)[0]

  ras1 = []
  decs1 = []
  ras2 = []
  decs2 = []

  for i in df_matched['RA FIRST(30s)'][ind_2_matches]:

    ras1.append(i[0])
    ras2.append(i[1])

  for j in df_matched['DEC FIRST(30s)'][ind_2_matches]:

    decs1.append(j[0])
    decs2.append(j[1])

  ras1 = np.array(ras1)
  decs1 = np.array(decs1)

  ras2 = np.array(ras2)
  decs2 = np.array(decs2)

  ras_manga = df_matched['RA'][ind_2_matches]
  decs_manga = df_matched['DEC'][ind_2_matches]

  c_1 = SkyCoord(ras1,decs1,frame = 'icrs', unit = 'degree')
  c_2 = SkyCoord(ras2,decs2,frame = 'icrs', unit = 'degree')
  c_manga = SkyCoord(ras_manga,decs_manga,frame = 'icrs', unit = 'degree')

  seps1 = c_manga.separation(c_1).value*3600
  seps2 = c_manga.separation(c_2).value*3600

  seps_near = []
  seps_far = []

  for i in range(len(seps1)):

    if seps1[i] < seps2[i]:

      seps_near.append(seps1[i])
      seps_far.append(seps2[i])
        
    else:

      seps_near.append(seps2[i])
      seps_far.append(seps1[i])

  seps_near = np.array(seps_near)
  seps_far = np.array(seps_far)

  ind_sources = np.where(df_matched['Radio Class'][ind_2_matches] == 2)

  return [seps_near, seps_far,ind_sources]


def get_first_3_seps(df_matched):

  ind_3_matches = np.where(df_matched['No. FIRST matches(30s)'] == 3)[0]

  ras1 = []
  decs1 = []
  ras2 = []
  decs2 = []
  ras3 = []
  decs3 = []

  for i in df_matched['RA FIRST(30s)'][ind_3_matches]:

    ras1.append(i[0])
    ras2.append(i[1])
    ras3.append(i[2])

  for j in df_matched['DEC FIRST(30s)'][ind_3_matches]:

    decs1.append(j[0])
    decs2.append(j[1])
    decs3.append(j[2])

  ras1 = np.array(ras1)
  decs1 = np.array(decs1)

  ras2 = np.array(ras2)
  decs2 = np.array(decs2)

  ras3 = np.array(ras3)
  decs3 = np.array(decs3)

  ras_manga = df_matched['RA'][ind_3_matches]
  decs_manga = df_matched['DEC'][ind_3_matches]

  c_1 = SkyCoord(ras1,decs1,frame = 'icrs', unit = 'degree')
  c_2 = SkyCoord(ras2,decs2,frame = 'icrs', unit = 'degree')
  c_3 = SkyCoord(ras3,decs3,frame = 'icrs', unit = 'degree')
  c_manga = SkyCoord(ras_manga,decs_manga,frame = 'icrs', unit = 'degree')

  seps1 = c_manga.separation(c_1).value*3600
  seps2 = c_manga.separation(c_2).value*3600
  seps3 = c_manga.separation(c_3).value*3600

  
  seps_near = []
  seps_mid = []

  for i in range(len(seps2)):

    temp_array = np.array([seps1[i],seps2[i],seps3[i]])
    temp_array = np.sort(temp_array)
    seps_mid.append(temp_array[1])
    seps_near.append(temp_array[0])

  seps_mid = np.array(seps_mid)
  seps_near = np.array(seps_near)
  

  ind_sources = np.where(df_matched['Radio Class'][ind_3_matches] == 2)[0]

  
  return [seps_near, seps_mid, ind_sources]



##############################################################

def load_pipe_data():

  '''Returns --> The reduced MaNGA catalog, MaNGA and radio catalogs of the matched objects.
  Input --> None

  Loads MaNGA Pipe3D data from the SDSS website and the required columns radio catalog from a personal cloud drive. 
  The MaNGA catalog is cleaned up by the removal of peculiar galaxies and galaxies that have the DAP and 
  MaNGA Quality Control bits set to "1". Matches objects in the two catalogs upto a maximum separation of 2 arcsec. 
  
  '''


  #url_pipe3d = 'https://data.sdss.org/sas/dr17/manga/spectro/pipe3d/v3_1_1/3.1.1/SDSS17Pipe3D_v3_1_1.fits'
  #path_pipe3d= download_file(url_pipe3d, cache= True)
  path_pipe3d = "../AGN_FIRST_files/data/MaNGA_catalog.fits"
  data_pipe3d_1 = fits.open(path_pipe3d)[1].data

  
  print("Original Length",len(data_pipe3d_1))

  inds_manga_doubles = [5275,1774,8691,6992,9045,6993,5006,5748,10041,5407,5893]
  data_pipe3d_all = np.delete(data_pipe3d_1, inds_manga_doubles)

  print("MaNGA doubles removed. Remaining length: ",len(data_pipe3d_all))

  '''
  indices_vd = np.where(data_pipe3d_all['vel_disp_ssp_cen'] < 65) 
  data_pipe3d_all = np.delete(data_pipe3d_all, indices_vd)

  print("Low Sigma removed. Remaining length: ",len(data_pipe3d_all))

  indices_vd_right = np.where(data_pipe3d_all['vel_disp_ssp_cen'] > 630) #should be 630
  data_pipe3d_all = np.delete(data_pipe3d_all, indices_vd_right)

  print("Very high sigma removed. Remaining length:",len(data_pipe3d_all))
  '''

  indices_SM_left = np.where(data_pipe3d_all['log_Mass'] < 8)
  data_pipe3d_all = np.delete(data_pipe3d_all, indices_SM_left)

  print("Low mass removed. Remaining length:",len(data_pipe3d_all))

  indices_SM_right = np.where(data_pipe3d_all['log_Mass'] > 12.5)
  data_pipe3d_all = np.delete(data_pipe3d_all, indices_SM_right)

  print("High mass removed. Remaining length:",len(data_pipe3d_all))

  indices_sfr_left = np.where(data_pipe3d_all['log_SFR_SF'] < -5.5)
  data_pipe3d_all = np.delete(data_pipe3d_all, indices_sfr_left)

  print("Low SFR removed. Remaining length:",len(data_pipe3d_all))

  indices_sfr_right = np.where(data_pipe3d_all['log_SFR_SF'] > 2.5)
  data_pipe3d_all = np.delete(data_pipe3d_all, indices_sfr_right)

  print("High SFR removed. Remaining length:",len(data_pipe3d_all))

  redshifts1 = data_pipe3d_all['nsa_redshift']
  data_pipe3d_all = np.delete(data_pipe3d_all , np.isnan(redshifts1))

  print("NaN redshifts removed. Remaining length:",len(data_pipe3d_all))

  data_pipe3d_all = np.delete(data_pipe3d_all , np.isnan(data_pipe3d_all['log_SFR_SF']))

  print("NaN SFR removed. Remaining length:",len(data_pipe3d_all))

  ssfr = np.log10(10**data_pipe3d_all['log_SFR_SF']/10**data_pipe3d_all["log_Mass"])
  ind_ssfr_left = np.where(ssfr < -16)

  data_pipe3d_all = np.delete(data_pipe3d_all, ind_ssfr_left)

  ssfr = np.log10(10**data_pipe3d_all['log_SFR_SF']/10**data_pipe3d_all["log_Mass"])
  ind_ssfr_right = np.where(ssfr > -9.5 )
  data_pipe3d_all = np.delete(data_pipe3d_all, ind_ssfr_right)

  print("sSFR outliers removed. Remaining length:",len(data_pipe3d_all))

  path_drp = "../AGN_FIRST_files/data/drpall-v3_1_1.fits"
  data_drp = fits.open(path_drp)[1].data

  plateifu_pipe = data_pipe3d_all['plateifu']
  plateifu_drp = data_drp['PLATEIFU']

  common_pifu, inds_pipe, inds_drp = np.intersect1d(plateifu_pipe,
  plateifu_drp,return_indices = True)

  data_pipe3d_all = data_pipe3d_all[inds_pipe]

  data_pipe3d_all = Table(data_pipe3d_all)

  data_pipe3d_all['ESWEIGHT'] = data_drp['ESWEIGHT'][inds_drp]
  data_pipe3d_all['MNGTARG3'] = data_drp['MNGTARG3'][inds_drp]

  indices_ancillary = np.where(data_pipe3d_all['MNGTARG3'] > 0)
  data_pipe3d_all = np.delete(data_pipe3d_all, indices_ancillary)

  print("Ancillary targets removed. Remaining length:",len(data_pipe3d_all))

  '''
  indices_esw = np.where(data_pipe3d_all['ESWEIGHT'] < 0)
  data_pipe3d_all = np.delete(data_pipe3d_all, indices_esw)

  print("Invalid ESWEIGHTs removed. Final length:",len(data_pipe3d_all))
  '''

  path_dap = "../AGN_FIRST_files/data/dapall-v3_1_1-3.1.0.fits"
  data_dap = fits.open(path_dap)[1].data

  plateifu_dap = data_dap['PLATEIFU']
  plateifu_pipe = data_pipe3d_all['plateifu']

  common_pifu1, inds_pipe1, inds_dap = np.intersect1d(plateifu_pipe,
  plateifu_dap,return_indices= True)

  data_pipe3d_all = data_pipe3d_all[inds_pipe1]
  data_pipe3d_all = Table(data_pipe3d_all)

  data_pipe3d_all['STELLAR_SIGMA_1RE'] = data_dap['STELLAR_SIGMA_1RE'][inds_dap]

  inds_low_dap_vd = np.where(data_pipe3d_all['STELLAR_SIGMA_1RE'] < 65)
  data_pipe3d_all = np.delete(data_pipe3d_all, inds_low_dap_vd)

  print("Low sigma galaxies removed. Remaining length:",len(data_pipe3d_all))

  inds_high_dap_vd = np.where(data_pipe3d_all['STELLAR_SIGMA_1RE'] > 630)
  data_pipe3d_all = np.delete(data_pipe3d_all, inds_high_dap_vd)

  print("High sigma galaxies removed. Remaining length:",len(data_pipe3d_all))


  '''
  ind_badifu = []

  for i in range(len(data_pipe3d_all)):
    if data_pipe3d_all['ellip'][i] > 0.75:
      ind_badifu.append(i)

  data_pipe3d_all = np.delete(data_pipe3d_all, ind_badifu)

  print("High ellipticity galaxies removed. Remaining length:",len(data_pipe3d_all))
  '''

  return data_pipe3d_all




def load_pipe_df():


  # load data
  data_pipe3d_all = load_pipe_data()

  #url_nvss = "https://urldefense.proofpoint.com/v2/url?u=https-3A__www.dropbox.com_s_lyuy4lp2lzi5p4d_CATALOG.FIT-3Fdl-3D0&d=DwMFaQ&c=slrrB7dE8n7gBJbeO0g-IQ&r=J9TPhCTsXLU5rvH37Iir6w&m=iqahptD4Y5INWgEYix4rQqPS-qYAZmwpc7Kj0bS2Que1p58myO0ZPWeMXhMgupDZ&s=TbiwgLzBNf8Rusy5ojMbZ8Gjhsc7DWPc0gOhJZvJasw&e="
  path_nvss = '../AGN_FIRST_files/data/nvss_full_catalog.FIT' 
  data_nvss = fits.open(path_nvss,memmap=True)[1].data 

  fints = get_integ_nvss_flux(data_nvss['MAJOR AX'], data_nvss['MINOR AX'], 45, data_nvss['PEAK INT'])
  data_nvss = Table(data_nvss)
  data_nvss['FINT'] = fints

  fints_error = get_fint_error(data_nvss['MAJOR AX'],data_nvss['MINOR AX'],45,data_nvss['PEAK INT'],data_nvss['I RMS'])
  data_nvss['FINT ERROR'] = fints_error

  #url_first = 'https://drive.google.com/uc?export=download&id=1itAIFo_i30j9kFxOGxIPDuxuvIJXyUNm'
  path_first = '../AGN_FIRST_files/data/FIRST.fits' 
  data_first = fits.open(path_first,memmap=True)[1].data

  c_nvss = SkyCoord(ra=data_nvss['RA(2000)']*u.degree, dec= data_nvss['DEC(2000)']*u.degree,
  frame= 'icrs', unit='deg')
  c_pipe3d = SkyCoord(ra=data_pipe3d_all['objra']*u.degree, dec= data_pipe3d_all['objdec']*u.degree,
  frame= 'icrs', unit='deg') 
  c_first = SkyCoord(ra=data_first['RA']*u.degree, dec= data_first['DEC']*u.degree,
  frame= 'icrs', unit='deg')

  #Match catalogs upto 3 arcmins
  idx_nvss, idx_pipe3d,d2d, d3d = c_pipe3d.search_around_sky(c_nvss, 3*u.arcmin)
  unique, counts = np.unique(idx_pipe3d, return_counts=True)  

  num_nvss_matches3m = np.zeros(len(data_pipe3d_all['log_SFR_SF']))
  num_nvss_matches3m[unique] = counts

  nvss_match_inds3m = np.zeros(len(data_pipe3d_all['log_SFR_SF']), dtype = 'object')
  nvss_match_inds3m = get_matching_inds(len(data_pipe3d_all['log_SFR_SF']),idx_pipe3d,idx_nvss)
  
  ##FIRST 3 arcmins

  idx_first, idx_pipe3d_first,d2d_first, d3d_first = c_pipe3d.search_around_sky(c_first,
  3*u.arcmin)
  unique_first, counts_first = np.unique(idx_pipe3d_first, return_counts=True)  
  num_first_matches3m = np.zeros(len(data_pipe3d_all['log_SFR_SF']))
  num_first_matches3m[unique_first] = counts_first

  first_match_inds3m = np.zeros(len(data_pipe3d_all['log_SFR_SF']), dtype = 'object')
  first_match_inds3m = get_matching_inds(len(data_pipe3d_all['log_SFR_SF']),idx_pipe3d_first,
  idx_first)

  # FIRST 30 arcsec

  idx_first_30s, idx_pipe3d_first_30s,d2d_first_30s, d3d_first_30s = c_pipe3d.search_around_sky(c_first,
  30*u.arcsec)
  unique_first_30s, counts_first_30s = np.unique(idx_pipe3d_first_30s, return_counts=True)  
  num_first_matches30s = np.zeros(len(data_pipe3d_all['log_SFR_SF']))
  num_first_matches30s[unique_first_30s] = counts_first_30s

  first_match_inds30s = np.zeros(len(data_pipe3d_all['log_SFR_SF']), dtype = 'object')
  first_match_inds30s = get_matching_inds(len(data_pipe3d_all['log_SFR_SF']),idx_pipe3d_first_30s,
  idx_first_30s)

  # Get RA and DEC of 3 arcmin matches
  ra_first3m = get_property_at_indices(data_first,first_match_inds3m,'RA')
  dec_first3m = get_property_at_indices(data_first,first_match_inds3m,'DEC')

  ra_first30s = get_property_at_indices(data_first,first_match_inds30s,'RA')
  dec_first30s = get_property_at_indices(data_first,first_match_inds30s,'DEC')

  ra_nvss3m = get_property_at_indices(data_nvss,nvss_match_inds3m,'RA(2000)')
  dec_nvss3m = get_property_at_indices(data_nvss,nvss_match_inds3m,'DEC(2000)')

  flux_first3m = get_property_at_indices(data_first,first_match_inds3m,'FINT')
  flux_first30s = get_property_at_indices(data_first,first_match_inds30s,'FINT')
  flux_nvss3m = get_property_at_indices(data_nvss,nvss_match_inds3m,'FINT')
  flux_nvss3m_error = get_property_at_indices(data_nvss,nvss_match_inds3m,'FINT ERROR')

  maj_ax_first30s = get_property_at_indices(data_first,first_match_inds30s,'MAJOR')
  pos_ang_first30s = get_property_at_indices(data_first,first_match_inds30s,'POSANG')

  sfr_pipe = data_pipe3d_all["log_SFR_SF"]
  ssfr_pipe = np.log10(10**data_pipe3d_all['log_SFR_SF']/10**data_pipe3d_all['log_Mass'])
  stel_mass_pipe = data_pipe3d_all["log_Mass"]
  z = data_pipe3d_all["nsa_redshift"]
  plateifu = data_pipe3d_all['plateifu']
  ra = data_pipe3d_all["objra"]
  dec = data_pipe3d_all["objdec"]
  vd_dap = np.log10(data_pipe3d_all['STELLAR_SIGMA_1RE'])
  ra_first = data_first['RA']
  dec_first = data_first['DEC']
  sfr_ssp = data_pipe3d_all["log_SFR_ssp"]
  log_N2_Ha = data_pipe3d_all["log_NII_Ha_cen"]
  log_S2_Ha = data_pipe3d_all["log_SII_Ha_cen"]
  log_O3_Hb = data_pipe3d_all["log_OIII_Hb_cen"]
  
  es_weight = data_pipe3d_all['ESWEIGHT']  
  ## To find matches with MaNGA in FIRST within 3 arcsec
  [inds_manga, inds_first] = match_manga_first(ra,dec,ra_first,dec_first,3)

  has_3arcsec_first_match = np.zeros(len(ra), dtype = bool)
  has_3arcsec_first_match[inds_manga] = np.ones(len(inds_manga),
  dtype = bool)
 
  galaxies_pipe = {"log_SFR_SF":sfr_pipe, 
  "log_Mass": stel_mass_pipe,
  "sSFR" : ssfr_pipe
  , "nsa_redshift": z, "RA": ra, "DEC": dec, "No. NVSS matches(3m)" : num_nvss_matches3m,
  "Indices NVSS matches(3m)" : nvss_match_inds3m,
  "3 arcsec FIRST match": has_3arcsec_first_match,
  "No. FIRST matches(3m)" : num_first_matches3m,
  "Indices FIRST matches(3m)": first_match_inds3m,"RA FIRST(3m)":ra_first3m,
  "DEC FIRST(3m)": dec_first3m,"RA NVSS(3m)":ra_nvss3m,
  "DEC NVSS(3m)":dec_nvss3m,"RA FIRST(30s)":ra_first30s,
  "DEC FIRST(30s)":dec_first30s,
  "Flux FIRST(3m)":flux_first3m,
  "Flux FIRST(30s)":flux_first30s,"Flux NVSS(3m)":flux_nvss3m,
  "No. FIRST matches(30s)": num_first_matches30s,"Maj FIRST(30s)": maj_ax_first30s,
  "PA FIRST(30s)":pos_ang_first30s, "Indices FIRST matches(30s)": first_match_inds30s
  ,"ESWEIGHT": es_weight, "plateifu": plateifu, "STELLAR_SIGMA_1RE": vd_dap,
  "log_SFR_ssp" : sfr_ssp,"Flux Error NVSS(3m)" : flux_nvss3m_error,
  "log_OIII_Hb_cen" : log_O3_Hb, "log_NII_Ha_cen" : log_N2_Ha,"log_SII_Ha_cen" : log_S2_Ha}

  df_pipe = pd.DataFrame(galaxies_pipe)

  return df_pipe

##############################################################


##############################################################
#Monte Carlo


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

def get_large(property , number):

  """ Returns a duplicated large array of a given array. Inputs the array and number of times the array is to be duplicated"""

  property_large = np.ones(len(property)*number)
    

  count = 0

  for i in np.arange(0,len(property_large),number):
    property_large[i:i+number] = property_large[i]*property[count]
    count = count+1
    
  return property_large

def get_L_Edd(vd):

  """Returns the limiting Eddington Luminosity of MaNGA galaxies without radio detection. Inputs the velocity dispersion"""
  log_mass_BH_all = np.float64(8.12+4.24*np.log10(10**vd/200)) 
  log_L_Edd_all = np.float64(np.log10(1.26e38)+log_mass_BH_all)

  return 10**log_L_Edd_all


class schechter(rv_continuous):  

  "Schechter distribution"

  def _cdf(self, x):

    if self.alpha < 1:
      self.norm = 1/gamma_integ(self.log_lambda_min,2.2,self.alpha,self.lambda_star)
      self.f = self.norm*gamma_integ(self.log_lambda_min,x,self.alpha,self.lambda_star)

      return self.f
    
    else:
      self.norm = 1/total_integral(self.log_lambda_min,2.2,self.alpha,self.lambda_star,self.correc_int_obj)
      self.f = self.norm*total_integral(self.log_lambda_min,x,self.alpha,self.lambda_star,self.correc_int_obj)

    return self.f
  
  def __init__(self,a,b,alpha_0,beta,lambda_star,log_lambda_min,prop,property_string): ##
      
    self.alpha_0 = alpha_0
    self.beta = beta
    self.lambda_star = lambda_star
    self.log_lambda_min = log_lambda_min
    self.prop = prop
    self.property_string = property_string
    self.a = a
    self.b = b
    self.alpha = alpha_prop(self.alpha_0,self.prop,self.beta,self.property_string)
    self.correc_int_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]

    super(schechter,self).__init__(a= self.a , b = self.b , name='schechter')

##############################################


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

  print("N galssssssssssssssssssssssssssss:::",size_prop_manga*frac_prop_manga)

  error_low = []
  error_high = []


  for i in range(nbins):

    exp = f_prop[i]*size_prop_manga[i]

    xlo = 0
    xhi = size_prop_manga[i]
    x = np.arange(xlo,xhi,1)


    cdf = betainc(size_prop_manga[i]-x,x+1,1-f_prop[i])
    indices_low = np.where(cdf > 0.16)


    if len(cdf) == 0:
      error_low.append(0.001)


    elif cdf[0] > 0.16:
      error_low.append(f_prop[i])
    
    else:
      error_low.append((exp-indices_low[0][0])/size_prop_manga[i])

    indices_high = np.where(cdf > 0.84)

    if len(indices_high[0]) == 0:
      error_high.append(0.001)

    elif indices_high[0][0] < exp:
      error_high.append(0)

    else:
      error_high.append((indices_high[0][0] - exp)/size_prop_manga[i])

    

  error_low = np.array(error_low)
  error_high = np.array(error_high)
  error_bar_prop = np.array([error_low,error_high])
  
  

  return [frac_prop_manga, prop_center_manga, f_prop, prop_center,error_bar_prop]


###################################################################################

def get_agn_frac_mbins(property,df_manga,dfs_mc, nbins, galaxy_type): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  if property == "log_Mass":
    bins = [10,10.5,11,11.5,12]

  if property == "STELLAR_SIGMA_1RE":
    bins = [1.75,2,2.25,2.5,2.75]

  if property == "log_SFR_SF":
    bins = [-5,-3.25,-1.5,0.25,2]
  
  if property == "sSFR":
    bins = [-16, -14.375, -12.75, -11.125, -9.5]

  inds_df_manga = np.where(df_manga['log_Mass'] > 10)[0]
  df_manga = df_manga.iloc[inds_df_manga]  

  if galaxy_type == 'sf':
    inds_galaxy_type = np.where(df_manga['sSFR'] > -11.5)
  
  if galaxy_type == 'q':
    inds_galaxy_type = np.where(df_manga['sSFR'] < -11.5)
  

  df_manga = df_manga.iloc[inds_galaxy_type]

  prop_center = []
  f_prop = []
  N_agn = np.zeros(nbins)
  N_all = np.zeros(nbins)

  for i in range(len(dfs_mc)):
    df_mc = dfs_mc[i]
    sorted_byprop_df = df_mc.sort_values(by = [property])
    binned_byprop_df = pd.cut(sorted_byprop_df[property], bins= bins)
    sorted_byprop_df['bin_prop'] = binned_byprop_df
    grped_byprop_df = sorted_byprop_df.groupby(["bin_prop"])
    j = 0

    for name, group in grped_byprop_df:
    
      string = str(name)
      split = string.split(',')
      left = float(split[0][1:])
      right = float(split[1][:-1])
      center = (left + right)/2

      if len(prop_center) < nbins:
        prop_center.append(center)

      N_agn[j] = N_agn[j] + group['agn flux detectable'].sum()
      N_all[j] = N_all[j] + len(group)
      j = j+1

  
  f_prop = N_agn/N_all
  prop_center = np.array(prop_center)

  sorted_byprop_manga = df_manga.sort_values(by = [property])

  binned_byprop_manga = pd.cut(sorted_byprop_manga[property], bins= bins)

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

  print("N galssssssssssssssssssssssssssss:::",size_prop_manga*frac_prop_manga)

  error_low = []
  error_high = []


  for i in range(nbins):

    exp = f_prop[i]*size_prop_manga[i]

    xlo = 0
    xhi = size_prop_manga[i]
    x = np.arange(xlo,xhi,1)

    cdf = betainc(size_prop_manga[i]-x,x+1,1-f_prop[i])
    indices_low = np.where(cdf > 0.16)

    if len(cdf) == 0:
      error_low.append(0.001)


    elif cdf[0] > 0.16:
      error_low.append(f_prop[i])
    
    else:
      error_low.append((exp-indices_low[0][0])/size_prop_manga[i])

    indices_high = np.where(cdf > 0.84)

    if len(indices_high[0]) == 0:
      error_high.append(0.001)

    elif indices_high[0][0] < exp:
      error_high.append(0)

    else:
      error_high.append((indices_high[0][0] - exp)/size_prop_manga[i])

    

  error_low = np.array(error_low)
  error_high = np.array(error_high)
  error_bar_prop = np.array([error_low,error_high])
  
  

  return [frac_prop_manga, prop_center_manga, f_prop, prop_center,error_bar_prop]

###################################################################################

def prob_agnness(mass,alpha_0,beta,lambda_star,log_lambda_min,log_lambda_c,correc_obj):
    
  alpha = alpha_prop(alpha_0,mass,beta,"log_Mass")

  if alpha < 1:

    prob = gamma_integ(log_lambda_c,2.2,alpha,
    lambda_star)/gamma_integ(log_lambda_min,2.2,alpha,lambda_star)

  if alpha > 1:

    prob = total_integral(log_lambda_c,2.2,alpha,
    lambda_star,correc_obj)/total_integral(log_lambda_min,2.2,alpha,lambda_star,correc_obj)

    
  return prob

def prob_agnness_unif(alpha_0,lambda_star,log_lambda_min,log_lambda_c,correc_obj):
    
  alpha = alpha_0

  if alpha <= 1:
    prob = gamma_integ(log_lambda_c,log_lambda_max,alpha,
    lambda_star)/gamma_integ(log_lambda_min,log_lambda_max,alpha,lambda_star)

  if alpha > 1:

    prob = total_integral(log_lambda_c,2.2,alpha,
    lambda_star,correc_obj)/total_integral(log_lambda_min,2.2,alpha,lambda_star,correc_obj)
  
  if prob > 1:
    prob = 1

  return prob



def get_absF_error(log_fracs):

  inds_valid = ~np.isnan(log_fracs)
  log_fracs = log_fracs[inds_valid]

  inds_valid = log_fracs > -np.inf

  log_fracs = log_fracs[inds_valid]
  log_median = np.percentile(log_fracs,50)

  log_frac_lower = np.percentile(log_fracs, 16)
  log_frac_upper = np.percentile(log_fracs, 84)

  error_lower = np.abs(log_median - log_frac_lower)
  error_upper = np.abs(log_frac_upper - log_median)

  return [log_median,error_lower, error_upper]


#########################################################################

def get_abs_agn_frac(model_parms,lambda_c, mass_low): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  nbins = 100

  [alpha_0,beta,lambda_star,log_lambda_min] = model_parms

  prop_center = []
  f_prop = []

  for i in np.linspace(mass_low,12,nbins):
    f_prop.append(prob_agnness(i,alpha_0,beta,lambda_star,log_lambda_min,lambda_c,correc_obj))
    prop_center.append(i)


  f_prop = np.array(f_prop)
  prop_center = np.array(prop_center)

  
  return [f_prop, prop_center]


def get_associated_nvss_indices(plateifu):

  if plateifu == '10510-9102':
      return [1,4]
  
  if plateifu == '11020-9101':
      return [0]
  
  if plateifu == '11024-12702':
      return [1]
  
  if plateifu == '11747-12703':
      return [2]
  
  if plateifu == '11747-12704':
      return [3]
  
  if plateifu == '11749-3702':
      return [1,2,4]
  
  if plateifu == '11827-3702':
      return [0]
  
  if plateifu == '11863-3703':
      return [2]
  
  if plateifu == '11863-6101':
      return [0]
  
  if plateifu == '11952-12703':
      return "None" #[0,1,2]
  
  if plateifu == '11960-9102':
      return [0,1]
  
  if plateifu == '12089-3702':
      return [3]
  
  if plateifu == '12090-6103':
      return [4]
  
  if plateifu == '12491-1901':
      return [2]
  
  if plateifu == '12620-12701':
      return "None"
  
  if plateifu == '12622-12702':
      return [1,2,3,4]
  
  if plateifu == '12624-6103':
      return [0,1,2,3]

  if plateifu == '7965-12704':
      return [0]
  
  if plateifu == '7980-12705':
      return 'None'
  
  if plateifu == '7981-12701':
      return [3]
  
  if plateifu == '7981-12702':
      return [0,1,2]
  
  if plateifu == '8092-3702':
      return "None"
  
  if plateifu == '8093-6101':
      return [0,2]
  
  if plateifu == '8245-1901':
      return "None"
  
  if plateifu == '8455-12703':
      return [3]
  
  if plateifu == '8481-3702':
      return [0,1,3]
  
  if plateifu == '8482-12704':
      return [0]
  
  if plateifu == '8592-12703':
      return [0]
  
  if plateifu == '8593-3704':
      return [1,2]
  
  if plateifu == '8600-9101':
      return [0]
  
  if plateifu == '8603-6101':
      return [1,2,4]

  if plateifu == '8612-1901':
      return [3]

  if plateifu == '8615-12704':
      return [3]

  if plateifu == '8616-6103':
      return [2]

  if plateifu == '8714-3704':
      return [3,4,7]

  if plateifu == '8717-6104':
      return [0,1]

  if plateifu == '8723-6103':
      return [3]

  if plateifu == '8943-12705':
      return [1,2,3]

  if plateifu == '8990-12702':
      return [0,1,2]

  if plateifu == '9024-1901':
      return [0,1,2]

  if plateifu == '9042-3702':
      return [0,1,2,3,4]

  if plateifu == '9043-3704':
      return [3]

  if plateifu == '9045-3701':
      return [0,2,3]    
  
  if plateifu == '9088-6103':
      return [0,1,2]
  
  if plateifu == '9089-12705':
      return [0]
  
  if plateifu == '9181-12702':
      return [0,2,3]
  
  if plateifu == '9182-12701':
      return [0]
  
  if plateifu == '9185-12703':
      return [0]
  
  if plateifu == '9485-12701':
      return "None"
  
  if plateifu == '9491-12701':
      return "None"
  
  if plateifu == '9503-3703':
      return [1,2]
  
  if plateifu == '9506-6103':
      return [2,3]
  
  if plateifu == '9884-9102':
      return [4]
  


def get_associated_first_indices(plateifu):

  if plateifu == '10510-9102':
      return [0]
  
  if plateifu == '11020-9101':
      return [0,2,3,4]
  
  if plateifu == '11024-12702':
      return [0,1,2]
  
  if plateifu == '11747-12703':
      return [0,1,2]
  
  if plateifu == '11747-12704':
      return [0,1,2,3]
  
  if plateifu == '11749-3702':
      return [0,1,2]
  
  if plateifu == '11827-3702':
      return [0,1,2,3]
  
  if plateifu == '11863-3703':
      return [0]
  
  if plateifu == '11863-6101':
      return [0,1,2,3]
  
  if plateifu == '11952-12703':
      return [0]
  
  if plateifu == '11960-9102':
      return [0,1,2,3]
  
  if plateifu == '12089-3702':
      return [0,1]
  
  if plateifu == '12090-6103':
      return [0,1]
  
  if plateifu == '12491-1901':
      return [0]
  
  if plateifu == '12620-12701':
      return "None"
  
  if plateifu == '12622-12702':
      return [0,1,2]
  
  if plateifu == '12624-6103':
      return [0]

  if plateifu == '7965-12704':
      return [0,1,2,3]
  
  if plateifu == '7980-12705':
      return 'None'
  
  if plateifu == '7981-12701':
      return "None"
  
  if plateifu == '7981-12702':
      return "None"
  
  if plateifu == '8092-3702':
      return "None"
  
  if plateifu == '8093-6101':
      return "None"
  
  if plateifu == '8245-1901':
      return "None"
  
  if plateifu == '8455-12703':
      return [0]
  
  if plateifu == '8481-3702':
      return [0,1,2,3,4]
  
  if plateifu == '8482-12704':
      return [0]
  
  if plateifu == '8592-12703':
      return [1,2,3]
  
  if plateifu == '8593-3704':
      return [0,1,2]
  
  if plateifu == '8600-9101':
      return [0,1,2,3]
  
  if plateifu == '8603-6101':
      return [0,1,2,3]

  if plateifu == '8612-1901':
      return [0]

  if plateifu == '8615-12704':
      return [0,1,2,3]

  if plateifu == '8616-6103':
      return [0]

  if plateifu == '8714-3704':
      return [0,1]

  if plateifu == '8717-6104':
      return [0]

  if plateifu == '8723-6103':
      return [0]

  if plateifu == '8943-12705':
      return "None"

  if plateifu == '8990-12702':
      return [0]

  if plateifu == '9024-1901':
      return [0,1,2,3]

  if plateifu == '9042-3702':
      return [0]

  if plateifu == '9043-3704':
      return [0]

  if plateifu == '9045-3701':
      return [0]    
  
  if plateifu == '9088-6103':
      return [0]
  
  if plateifu == '9089-12705':
      return [0,1,2]
  
  if plateifu == '9181-12702':
      return [0,1,2]
  
  if plateifu == '9182-12701':
      return [0,1]
  
  if plateifu == '9185-12703':
      return [0,1,2]
  
  if plateifu == '9485-12701':
      return "None"
  
  if plateifu == '9491-12701':
      return "None"
  
  if plateifu == '9503-3703':
      return [0]
  
  if plateifu == '9506-6103':
      return [0]
  
  if plateifu == '9884-9102':
      return "None"
  

def delete_unrelated_nvss_flux_vi(df_matched, index):

  if type(get_radio_class(df_matched,index)) ==  str:
    

    if is_vi_source(df_matched,index) == True:

      plateifu = df_matched['plateifu'][index]
      associated_indices = get_associated_nvss_indices(plateifu)
      df_matched["Flux NVSS(3m)"].iloc[index] = np.array(df_matched["Flux NVSS(3m)"].iloc[index])[associated_indices]
      df_matched["RA NVSS(3m)"].iloc[index] = np.array(df_matched["RA NVSS(3m)"].iloc[index])[associated_indices]
      df_matched["DEC NVSS(3m)"].iloc[index] = np.array(df_matched["DEC NVSS(3m)"].iloc[index])[associated_indices]
      df_matched["Indices NVSS matches(3m)"].iloc[index] = np.array(df_matched["Indices NVSS matches(3m)"].iloc[index])[associated_indices]
      df_matched["No. NVSS matches(3m)"].iloc[index] = len(associated_indices)

      if "Flux Error NVSS(3m)" in df_matched.columns:

        df_matched["Flux Error NVSS(3m)"].iloc[index] = np.array(df_matched["Flux Error NVSS(3m)"].iloc[index])[associated_indices]



      associated_first_indices = get_associated_first_indices(plateifu)

      if type(associated_first_indices) != str:

        df_matched["Flux FIRST(30s)"].iloc[index] = np.array(df_matched["Flux FIRST(30s)"].iloc[index])[associated_first_indices]
        df_matched["RA FIRST(30s)"].iloc[index] = np.array(df_matched["RA FIRST(30s)"].iloc[index])[associated_first_indices]
        df_matched["DEC FIRST(30s)"].iloc[index] = np.array(df_matched["DEC FIRST(30s)"].iloc[index])[associated_first_indices]
        df_matched["Indices FIRST matches(30s)"].iloc[index] = np.array(df_matched["Indices FIRST matches(30s)"].iloc[index])[associated_first_indices]
        df_matched["No. FIRST matches(30s)"].iloc[index] = len(associated_first_indices)


def is_vi_source(df_matched,index):

  plateifu = df_matched['plateifu'][index]

  if type(get_associated_nvss_indices(plateifu)) == str:
    return False
  
  if type(get_associated_nvss_indices(plateifu)) == list:

    return True
  


def classify_vi_source(df_manga,index):

  if is_vi_source(df_manga,index) == True:

    flux_nvss = df_manga['Flux NVSS(3m)'][index]
    flux_nvss = np.array(flux_nvss).sum()
    n_nvss = df_manga['No. NVSS matches(3m)'][index]
    n_first = df_manga['No. FIRST matches(30s)'][index]


    if flux_nvss > 2.5:
      if n_nvss == 1 and n_first == 1:
        return 1
    
      elif n_nvss == 1 and n_first > 1:
        return 2
    
      elif n_nvss == 1 and n_first == 0:
        return 3
    
      elif n_nvss > 1:
        return 4
      
      else:
        return 0
      
    else:
      return 0
  
  else:
    return 0
  
###############################################################################
  
def get_fint_error(theta_M,theta_m,theta_N,peak, rms_pixel_noise):


  fint = get_integ_nvss_flux(theta_M,theta_m,theta_N,peak)

  sigma2_A = get_sigma2_A(theta_M,theta_m,theta_N,peak, rms_pixel_noise)
  sigma2_M = get_sigma2_M(theta_M,theta_m,theta_N,peak, rms_pixel_noise)
  sigma2_m = get_sigma2_m(theta_M,theta_m,theta_N,peak, rms_pixel_noise)


  theta_M = theta_M*3600
  theta_m = theta_m*3600
  peak = peak*1000
  rms_pixel_noise = rms_pixel_noise*1000

  first_term = fint**2*((sigma2_A/peak**2) + (theta_N**2/(theta_M*theta_m))*( (sigma2_M/theta_M**2)**2 + (sigma2_m/theta_m**2)**2))
  second_term = (fint*0.03)**2
  third_term = 0.0003**2

  sum = first_term + second_term + third_term

  sqrt_sum = sum**0.5

  return sqrt_sum


def get_nvss_flux_error(df_matched,index):
  if get_radio_class(df_matched,index) != 0:

    if type(get_radio_class(df_matched,index)) != str:

      flux_array = np.array(df_matched['Flux NVSS(3m)'][index])

      flux = flux_array.sum()
      
      if flux > 2.5:

        flux_error_array = np.array(df_matched['Flux Error NVSS(3m)'][index])
        flux_error = ((flux_error_array**2).sum())**0.5

        return flux_error

      else:
        return -999


    elif type(get_radio_class(df_matched,index)) == str:

      if is_vi_source(df_matched,index) == True:
        flux_array = np.array(df_matched['Flux NVSS(3m)'][index])
        flux = flux_array.sum()
      
        if flux > 2.5:
            flux_error_array = np.array(df_matched['Flux Error NVSS(3m)'][index])
            flux_error = ((flux_error_array**2).sum())**0.5
            return flux_error

        else:
          return -999
      
      else:
        return -999
    
    else:
      return -999
  
  else:
    return -999
    


def flatten_array(array, datatype):
    
    max_length = 0

    for i in array:
        if isinstance(i, np.ndarray) or isinstance(i, list):
            if len(i) > max_length:
                max_length = len(i)

    new_array = -999*np.ones((len(array), max_length), dtype=datatype)

    for i in range(len(array)):
        if isinstance(array[i], np.ndarray):
            new_array[i, :len(array[i])] = array[i]

        elif isinstance(array[i], list):
            new_array[i, :len(array[i])] = np.array(array[i])
        else:
            new_array[i] = -999 * np.ones(max_length, dtype=datatype)

    return new_array



def get_snr_rho2(theta_M,theta_m,theta_N, peak,alpha_M, alpha_m, rms_pixel_noise):

  theta_M = theta_M*3600
  theta_m = theta_m*3600
  peak = peak*1000
  rms_pixel_noise = rms_pixel_noise*1000

  coeff = theta_m*theta_M*peak**2/(4*rms_pixel_noise*theta_N**2)
  major = (1 + (theta_N/theta_M)**2)**alpha_M
  minor = (1 + (theta_N/theta_m)**2)**alpha_m

  product = coeff*minor*major

  return product


def get_integ_nvss_flux(theta_M,theta_m,theta_N,peak):

  """Input theta_M, theta_m in degrees. Theta_N in arcsecs. peak in Jy"""

  theta_M = theta_M*3600
  theta_m = theta_m*3600
  peak = peak*1000

  fint = peak*(theta_M*theta_m/(theta_N**2))

  return fint


def get_sigma2_M(theta_M,theta_m,theta_N,peak,rms_pixel_noise):

  theta_M = theta_M*3600
  theta_m = theta_m*3600
  peak = peak*1000
  rms_pixel_noise = rms_pixel_noise*1000

  alpha_M = 5/2
  alpha_m = 1/2

  rho2 = get_snr_rho2(theta_M,theta_m,theta_N, peak,alpha_M, alpha_m, rms_pixel_noise)

  first_term = 2*theta_M**2/rho2
  sec_term = (0.02*theta_N)**2

  sum = first_term + sec_term

  return sum


def get_sigma2_m(theta_M,theta_m,theta_N,peak,rms_pixel_noise):

  theta_M = theta_M*3600
  theta_m = theta_m*3600
  peak = peak*1000
  rms_pixel_noise = rms_pixel_noise*1000

  alpha_M = 1/2
  alpha_m = 5/2

  rho2 = get_snr_rho2(theta_M,theta_m,theta_N, peak,alpha_M, alpha_m, rms_pixel_noise)

  first_term = 2*theta_m**2/rho2
  sec_term = (0.02*theta_N)**2

  sum = first_term + sec_term

  return sum


def get_sigma2_A(theta_M,theta_m,theta_N,peak, rms_pixel_noise):

  theta_M = theta_M*3600
  theta_m = theta_m*3600
  peak = peak*1000
  rms_pixel_noise = rms_pixel_noise*1000

  alpha_M = 3/2
  alpha_m = 3/2

  rho2 = get_snr_rho2(theta_M,theta_m,theta_N,peak,alpha_M, alpha_m, rms_pixel_noise)

  sig2_A = ((2*peak**2)/(rho2)) + (peak*0.03)**2 + (0.0003)**2

  return sig2_A



##################################

def get_chi2(y_hat,y,sigma):

  f = (y_hat - y)**2/sigma**2

  if type(f) != np.ndarray:
    f = np.array(f)
    
  chi_sq = f

  return chi_sq

#######################################

def get_mdot_edd(vd, epsilon, c):

  log_mass_BH = np.float64(8.12+4.24*np.log10(10**vd/200))  # log(M_bh/Msun)
  m_dot_edd = 1.26e38*(10**log_mass_BH)/(epsilon*c**2)

  return m_dot_edd


def get_ER_nicole_thomas(log_Lr, vd,epsilon,c):

  coef = 4e17
  term1 = (10**log_Lr/1e30)**(12/17)

  mdot_edd = get_mdot_edd(vd,epsilon,c)

  total = coef*term1/mdot_edd
  
  return total



def p_cav(log_Lr):

  log_pcav = 0.75*log_Lr + 13.91

  return log_pcav


def get_er_pcav(Lr, vd):

  pcav = p_cav(Lr)
  log_mass_BH = np.float64(8.12+4.24*np.log10(10**vd/200))  # log(M_bh/Msun)
  log_L_Edd= np.float64(np.log10(1.26e38)+log_mass_BH)
  L_Edd= 10**log_L_Edd

  er = 10**pcav/L_Edd

  return er


def get_er_pcav_check_agn(Lr, vd, agnness):

  if agnness == True:
    pcav = p_cav(Lr)
    log_mass_BH = np.float64(8.12+4.24*np.log10(10**vd/200))  # log(M_bh/Msun)
    log_L_Edd= np.float64(np.log10(1.26e38)+log_mass_BH)
    L_Edd= 10**log_L_Edd

    er = 10**pcav/L_Edd
  
  else: 
    er = 1e-99

  return er

vector_er_pcav_check_agn = np.vectorize(get_er_pcav_check_agn)


def load_matched_df():

  df_pipe = load_pipe_df()

  for i in range(len_manga_Ha):
    delete_unrelated_flux_2_1(df_pipe,i)

  for i in range(len_manga_Ha):
    delete_unrelated_flux_3_1(df_pipe,i)

  for i in range(len_manga_Ha):
    delete_unrelated_nvss_flux_vi(df_pipe,i)
  
  radio_class = []

  for i in range(len_manga_Ha):
    if type(get_radio_class(df_pipe,i)) == str:

      if is_vi_source(df_pipe, i) == True:
        radio_class.append(classify_vi_source(df_pipe,i))
      
      else:
        radio_class.append(0)
    
    else:
      radio_class.append(get_radio_class(df_pipe,i))


  radio_class = np.array(radio_class, dtype = object)

  df_pipe['Radio Class'] = radio_class

  flux = []

  for i in range(len_manga_Ha):
    flux.append(get_nvss_flux(df_pipe,i))

  flux = np.array(flux)
  df_pipe['Flux'] = flux

  flux_error = []

  for i in range(len_manga_Ha):
    flux_error.append(get_nvss_flux_error(df_pipe,i))

  flux_error = np.array(flux_error)
  df_pipe['fint_error'] = flux_error

  plt.hist(flux)
  plt.show()

  sigma = df_pipe['STELLAR_SIGMA_1RE'].to_numpy()
  z = df_pipe['nsa_redshift'].to_numpy()
  Lr = get_radio(flux,z)
  sfr = df_pipe['log_SFR_SF'].to_numpy()
  ER = get_ER(sigma,Lr)
  ER_nt = get_ER_nicole_thomas(Lr,sigma,0.1,3e10)

  ER_cav = get_er_pcav(Lr,sigma)

  agn_flux_detectable = vector_flux_detected(sfr,Lr,z,check_agn = True)
  flux_detectable = vector_flux_detected(sfr,Lr,z,check_agn = False)
  is_agn = vector_is_agn(sfr,Lr)

  agn_flux_detectable_plain = vector_flux_detected_plain_cut(sfr,Lr,z,check_agn = True)
  is_agn_plain = vector_is_agn_plain_cut(sfr,Lr)

  df_pipe["Lr"] = Lr
  df_pipe['agn flux detectable'] = agn_flux_detectable
  df_pipe['ER'] = ER
  df_pipe['ER_nt'] = ER_nt
  df_pipe['ER_cav'] = ER_cav

  df_pipe['flux detectable'] = flux_detectable
  detected_nagn = flux_detectable
  detected_nagn[is_agn] = np.zeros(len(detected_nagn[is_agn]), dtype = bool)
  df_pipe['detected_nagn'] = detected_nagn

  df_pipe['is agn'] = is_agn

  df_pipe['agn flux detectable plain'] = agn_flux_detectable_plain
  df_pipe['is agn plain'] = is_agn_plain

  return df_pipe


def get_AGN_lim_plain(sf,L_lim):

  """Returns the limiting luminosity of an AGN in the MaNGA galaxies
  without matches in the radio catalog."""

  SF_radio_lim =  Lsfr_interp(sf) + 1 # We're providing a buffer of 1 dex above the regressed line before classifying a galaxy as an AGN.
                                          
  L_lim_modified = L_lim
  if SF_radio_lim > L_lim_modified:
    L_lim_modified = SF_radio_lim

  return L_lim_modified

v_get_AGN_lim_plain = np.vectorize(get_AGN_lim_plain)


def get_ER_lim_agn_cav_plain(vd,z,sfr):
  
  """  Returns the limiting AGN Eddington Ratio of MaNGA galaxies without radio detection. 
  Inputs the velocity dispersion and redshift of the galaxies."""

  lumi_lim = get_L_lim(z)
  lumi_lim = v_get_AGN_lim_plain(sfr,lumi_lim)
  lambda_lim = get_er_pcav(lumi_lim,vd)

  return lambda_lim


def get_ER_lim_agn_cav(vd,z,sfr):
  
  """  Returns the limiting AGN Eddington Ratio of MaNGA galaxies without radio detection. 
  Inputs the velocity dispersion and redshift of the galaxies."""

  lumi_lim = get_L_lim(z)
  lumi_lim = v_get_AGN_lim(sfr,lumi_lim)
  lambda_lim = get_er_pcav(lumi_lim,vd)

  return lambda_lim


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

  print("The min isss::::",lambda_lim_non_agn.min())

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

    #print(ln_likelihood)

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

    #print(lambda_lim_non_agn.min())

    ln_likelihood = ln_likelihood + (volw_non_agn_alps_gt_1*np.log(
    likelihoods_alps_gt_1)).sum() + (volw_non_agn_alps_lt_1*np.log(likelihoods_alps_lt_1)).sum()

    #print(ln_likelihood)

    return -ln_likelihood

  from scipy.optimize import minimize

  alpha0_start = 1.5
  beta_start = -3
  lambda_star_start = 0.001
  min_start = -5

  theta_start = [alpha0_start,beta_start,lambda_star_start,min_start]
  result = minimize(ln_p_sch,theta_start, method= 'Nelder-Mead',
  bounds = [(None,None),(None,None),(None,None),(-7,-3)])


  return [result.x[0], result.x[1], result.x[2], result.x[3]]


def get_Lr_nt(er,vd,epsilon,c):

  first_term = 1e30*(er/4e17)**(17/12)
  mdot_edd_term = (get_mdot_edd(vd,epsilon,c))**(17/12)

  total = np.log10(first_term*mdot_edd_term)

  return total


def get_Lr_pcav(pcav):

  log_Lr = (pcav - 13.91)/0.75

  return log_Lr


def get_agn_frac_sfq(property,df_manga): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  nbins = 8 #was 8

  sorted_byprop_manga = df_manga.sort_values(by = [property])

  binned_byprop_manga = pd.cut(sorted_byprop_manga[property], bins= nbins)

  sorted_byprop_manga['bin prop'] = binned_byprop_manga

  grped_byprop_manga = sorted_byprop_manga.groupby(["bin prop"])

  prop_center_manga = []
  frac_prop_manga = []
  size_prop_manga = []

  frac_sf = []
  frac_q = []

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

    condition_sf = group['sSFR'] > -11.25
    average_sf_manga = group["agn flux detectable"][condition_sf].mean()
    frac_sf.append(average_sf_manga)

    print(len(group["agn flux detectable"][condition_sf]))

    condition_q = group['sSFR'] < -11.25
    average_q_manga = group["agn flux detectable"][condition_q].mean()
    frac_q.append(average_q_manga)

    print(len(group["agn flux detectable"][condition_q]))



  size_prop_manga = np.array(size_prop_manga)
  frac_prop_manga = np.array(frac_prop_manga)
  prop_center_manga = np.array(prop_center_manga)
  frac_q = np.array(frac_q)
  frac_sf = np.array(frac_sf)
  

  return [frac_prop_manga, prop_center_manga, frac_q , frac_sf]



def plot_frac_sfq(property, df_manga):

  [frac_prop_manga, prop_center_manga, frac_q , frac_sf] = get_agn_frac_sfq(
  property,df_manga)
  
  prop_center_manga_finite_sf = prop_center_manga[frac_sf > 10**-4]
  frac_sf_finite = frac_sf[frac_sf > 10**-5]
  
  prop_center_manga_finite_q = prop_center_manga[frac_q > 10**-4]
  frac_q_finite = frac_q[frac_q > 10**-5]

  #plt.scatter(prop_center_manga, frac_prop_manga, label = "Observation", c = 'k', s = 1.5*180)
  plt.scatter(prop_center_manga_finite_sf, frac_sf_finite, label = "Detected AGN fraction (Star Forming)", c = 'royalblue', s = 1.5*180, alpha = 0.6)
  plt.scatter(prop_center_manga_finite_q, frac_q_finite, label = "Detected AGN fraction (Quenched)", c = 'tab:red', s = 1.5*180, alpha= 0.6)
  plt.plot(prop_center_manga_finite_q, frac_q_finite, c = 'tab:red', linewidth = 2)
  plt.plot(prop_center_manga_finite_sf, frac_sf_finite, c = 'royalblue', linewidth = 2)
  plt.legend(prop = {'size' : 1.5*15})
  plt.yscale('log')
  plt.ylim(7e-6,1.5)
  plt.yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1], labels = [-5,-4,-3,-2,-1,0])
  plt.tick_params(axis ='y',labelsize =30)
  plt.minorticks_off()
  #plt.ylabel(r"$\log_{\rm 10} ~\rm f_{\rm AGN}$", fontsize = 1.2*30)


##############################################################################################################

def group_by_mass(df,nbins,galaxy_type):

  bins = [10,10.4,10.8,11.2,11.6,12]

  sorted_byprop_df = df.sort_values(by = ['log_Mass'])
  binned_byprop_df = pd.cut(sorted_byprop_df['log_Mass'], bins= bins)
  sorted_byprop_df['Mass_bin'] = binned_byprop_df
  grped_byprop_df = sorted_byprop_df.groupby(["Mass_bin"])

  j = 1
  for name, group in grped_byprop_df:
    
    print(name)
    group.to_pickle("../AGN_FIRST_files/data/df_M_ssfr_bins/df_" + galaxy_type + str(j) + ".pkl")

    if j < nbins:
      j =  j+1
    
    else:
      break



def mle_msf_bins(df_msf_bin,correction_integ_obj):

  '''Returns --> Model parameters alpha, beta, lamda_star and log_lambda_min after running MLE. 

  Inputs --> 1) A string specifying the model type ("log_Mass" for the mass model) 
  2) A pandas dataframe of properties of all galaxies. This can be generated using "load_matched_df()".
  3) A file containing a grid of interpolations of integrals of the power law correction term that turns the power law into a Schecter function.
  Use the function "correction_interp()" to generate the grid and then save it as a file that can be input here. np.save(, allow_pickle = True) works
  well to save the file.
  4) Set use_volw to True if you want to include the Wake weights while fitting. Set it to False otherwise. 
  
  '''
  mass = df_msf_bin['log_Mass']
  if type(mass) != np.ndarray:
    mass = mass.to_numpy()

  Lr = df_msf_bin['Lr']
  if type(Lr) != np.ndarray:
    Lr = Lr.to_numpy()

  sfr = df_msf_bin['log_SFR_SF']
  if type(sfr) != np.ndarray:
    sfr = sfr.to_numpy()

  vd = df_msf_bin['STELLAR_SIGMA_1RE']
  if type(vd) != np.ndarray:
    vd = vd.to_numpy()

  z = df_msf_bin['nsa_redshift']
  if type(z) != np.ndarray:
    z = z.to_numpy()

  ind_agn = df_msf_bin["agn flux detectable"].to_numpy(dtype= bool)

  Lr_agn = Lr[ind_agn]
  vd_dap_agn = vd[ind_agn]

  sfr_non_agn = np.delete(sfr, ind_agn)
  vd_non_agn = np.delete(vd, ind_agn)
  z_non_agn = np.delete(z, ind_agn)

  lambda_lim_non_agn = get_ER_lim_agn_cav(vd_non_agn,z_non_agn,sfr_non_agn)
  lambda_agn = get_er_pcav(Lr_agn,vd_dap_agn)

  def ln_p_msf_bins(theta_now): 

    alpha_0 = theta_now[0]
    lambda_star = 10**theta_now[1]
    log_lambda_min = theta_now[2]

    if (alpha_0 < -10) or (alpha_0 > 10):
      return -np.inf

    if (lambda_star <= 0) or (lambda_star > 1):
      return -np.inf
    
    if log_lambda_min < -10:
      return -np.inf
  
    if log_lambda_min > np.log10(lambda_lim_non_agn.min()):
      return -np.inf

    if len(lambda_agn) > 0:
      if log_lambda_min > np.log10(lambda_agn.min()):
        return -np.inf

    if alpha_0 > 1:
      norm_agn = 1/(total_integral(log_lambda_min,log_lambda_max,alpha_0,
      lambda_star,correction_integ_obj)) 
      norms_agn = norm_agn*np.ones(len(lambda_agn))


    if alpha_0 <= 1:
      norm_agn = 1/(gamma_integ(log_lambda_min,log_lambda_max,alpha_0,
      lambda_star)) 
      norms_agn = norm_agn*np.ones(len(lambda_agn))


    likelihoods_agn = norms_agn*schechter_prop_mcmc_mbins(lambda_agn,
    alpha_0,lambda_star)

    ln_likelihood = np.log(likelihoods_agn).sum()
    
    alphas_non_agn = alpha_0*np.ones(len(lambda_lim_non_agn))
    norms_non_agn = np.ones(len(alphas_non_agn))

    if alpha_0 > 1:

      norms_non_agn = 1/(total_integral(log_lambda_min,log_lambda_max,alphas_non_agn,
      lambda_star,correction_integ_obj))


      likelihoods = np.minimum(norms_non_agn*total_integral(log_lambda_min,
      np.log10(lambda_lim_non_agn),alphas_non_agn,lambda_star,correction_integ_obj), 1)


    if alpha_0 <= 1:
      norm_non_agn = 1/(gamma_integ(log_lambda_min,log_lambda_max,alphas_non_agn,
      lambda_star))

      norms_non_agn = norm_non_agn*np.ones(len(lambda_lim_non_agn))
      likelihoods = norms_non_agn*gamma_integ(log_lambda_min,
      np.log10(lambda_lim_non_agn)
      ,alphas_non_agn,lambda_star)
      

    ln_likelihood = ln_likelihood + np.log(likelihoods).sum()

    if np.isnan(ln_likelihood):
      return -np.inf
    

    print(ln_likelihood)

    if ln_likelihood == 0:
      print(theta_now)

    return -ln_likelihood

  from scipy.optimize import minimize

  alpha0_start = 1.5
  lambda_star_start = -2
  min_start = -6.5

  log_lambda_lim_non_agn_min = np.log10(lambda_lim_non_agn.min())

  theta_start = [alpha0_start,lambda_star_start,min_start]
  result = minimize(ln_p_msf_bins,theta_start, method= 'Nelder-Mead',
  bounds = [(-10,10),(-9,1),(-10,log_lambda_lim_non_agn_min)])


  return [result.x[0], result.x[1], result.x[2]]


def mle_bins(galaxy_type, nbins, correc_integ_obj):

  for i in range(nbins):
    bin_num = i + 1
    path = "../AGN_FIRST_files/data/df_M_ssfr_bins/df_" + galaxy_type + str(bin_num) + ".pkl"
    df = pd.read_pickle(path)

    M = 0.5*(df['log_Mass'].max() + df['log_Mass'].min())

    model_para = mle_msf_bins(df,correc_integ_obj)
    alpha_0, lambda_star, log_lambda_min = model_para

    np.savez_compressed('../AGN_FIRST_files/results/MLE/mass_ssfr_bins/mle_' + galaxy_type + str(bin_num) + '.npz', M = M,
    alpha_0 = alpha_0,lambda_star = lambda_star,log_lambda_min = log_lambda_min)



def get_abs_agn_frac_mbins(model_parms, log_lambda_c):

  M = model_parms[0]
  alpha_0 = model_parms[1]
  lambda_star = model_parms[2]
  log_lambda_min = model_parms[3]

  F = prob_agnness_unif(alpha_0, lambda_star, log_lambda_min, log_lambda_c, correc_obj)
  log_F = np.log10(F)
  
  return [M, log_F]


def get_absFrac_withebars_mbins(model_parms, mcmc_samples, log_lambda_c):

  [M, log_F] = get_abs_agn_frac_mbins(model_parms,log_lambda_c)

  alpha0s = mcmc_samples[:,0]
  lamstars = 10**mcmc_samples[:,1]
  logmins = mcmc_samples[:,2]

  log_all_fracs_in_bin = []

  for i in range(len(logmins)):

    log_all_fracs_in_bin.append(get_abs_agn_frac_mbins(
    [M,alpha0s[i],lamstars[i],logmins[i]],log_lambda_c)[1])
  
  log_all_fracs_in_bin = np.array(log_all_fracs_in_bin)

  lbs = []
  ubs = []

  log_median_F,error_low, error_high = get_absF_error(log_all_fracs_in_bin)

  lbs.append(error_low)
  ubs.append(error_high)

  lbs = np.array(lbs)
  ubs = np.array(ubs)

  ebars = np.array([lbs,ubs])

  return [M,log_median_F,ebars]



def schechter_prop_mcmc_mbins(lamda,alpha_0, lambda_star):

  """This is derived from the true model. This function is used while integrating the
  model, as it allows us to vary lambda in log units. This is used only in the Monte Carlo stage of
  the analysis"""

  
  p_Xi= np.log(10)*lambda_star*(lamda/lambda_star)**(-alpha_0+1)*np.exp(-lamda/lambda_star)
   
  return p_Xi


def ln_p_sch_mcmc(df_msf_bin,theta_now,correction_integ_obj): 

  alpha_0 = theta_now[0]
  lambda_star = 10**theta_now[1]
  log_lambda_min = theta_now[2]


  if (alpha_0 < -10) or (alpha_0 > 10):
    return -np.inf

  if (lambda_star <= 0) or (lambda_star > 1):
    return -np.inf
  
  if log_lambda_min < -10:
    return -np.inf
  
  if alpha_0 == 1:
    return -np.inf
  
  mass = df_msf_bin['log_Mass']
  if type(mass) != np.ndarray:
    mass = mass.to_numpy()

  vol_weights = np.ones(len(df_msf_bin['ESWEIGHT']))
  if type(vol_weights) != np.ndarray:
    vol_weights = vol_weights.to_numpy()

  Lr = df_msf_bin['Lr']
  if type(Lr) != np.ndarray:
    Lr = Lr.to_numpy()

  sfr = df_msf_bin['log_SFR_SF']
  if type(sfr) != np.ndarray:
    sfr = sfr.to_numpy()
  

  vd = df_msf_bin['STELLAR_SIGMA_1RE']
  if type(vd) != np.ndarray:
    vd = vd.to_numpy()
  
  z = df_msf_bin['nsa_redshift']
  if type(z) != np.ndarray:
    z = z.to_numpy()

  ind_agn = df_msf_bin["agn flux detectable"].to_numpy(dtype= bool)

  Lr_agn = Lr[ind_agn]
  vd_dap_agn = vd[ind_agn]
  
  sfr_non_agn = np.delete(sfr, ind_agn)
  vd_non_agn = np.delete(vd, ind_agn)
  z_non_agn = np.delete(z, ind_agn)

  lambda_lim_non_agn = get_ER_lim_agn_cav(vd_non_agn,z_non_agn,sfr_non_agn)
  lambda_agn = get_er_pcav(Lr_agn,vd_dap_agn)


  if log_lambda_min > np.log10(lambda_lim_non_agn.min()):
    return -np.inf
  
  if len(lambda_agn) > 0:
    if log_lambda_min > np.log10(lambda_agn.min()):
      return -np.inf

    
  alphas_agn = alpha_0*np.ones(len(lambda_agn))

  if alpha_0 > 1:
    norm_agn = 1/(total_integral(log_lambda_min,log_lambda_max,alphas_agn,
    lambda_star,correction_integ_obj)) 
    norms_agn = norm_agn*np.ones(len(alphas_agn))

  if alpha_0 < 1:
    norm_agn = 1/(gamma_integ(log_lambda_min,log_lambda_max,alphas_agn,
    lambda_star)) 
    norms_agn = norm_agn*np.ones(len(alphas_agn))

  likelihoods_agn = norms_agn*schechter_prop_mcmc_mbins(lambda_agn,
  alpha_0,lambda_star)

  ln_likelihood = np.log(likelihoods_agn).sum()

  alphas_non_agn = alpha_0*np.ones(len(lambda_lim_non_agn))
  norms_non_agn = np.ones(len(alphas_non_agn))

  if alpha_0 > 1:
    norm_non_agn = 1/(total_integral(log_lambda_min,log_lambda_max,alphas_non_agn,
    lambda_star,correction_integ_obj))

    norms_non_agn = norm_non_agn*np.ones(len(lambda_lim_non_agn))

    likelihoods = np.minimum(norms_non_agn*total_integral(log_lambda_min,
    np.log10(lambda_lim_non_agn),alphas_non_agn,lambda_star,correction_integ_obj), 1)


  if alpha_0 < 1:
    norm_non_agn = 1/(gamma_integ(log_lambda_min,log_lambda_max,alphas_non_agn,
    lambda_star))

    norms_non_agn = norm_non_agn*np.ones(len(lambda_lim_non_agn))
    likelihoods = norms_non_agn*gamma_integ(log_lambda_min,
    np.log10(lambda_lim_non_agn)
    ,alphas_non_agn,lambda_star)

  ln_likelihood = ln_likelihood + np.log(likelihoods).sum()

  print(ln_likelihood)

  return ln_likelihood


def log_prior(theta_now):

  alpha_0, log_lambda_star, log_lambda_min = theta_now

  if -10 < alpha_0 < 10 and -10 < log_lambda_star < 1 and -10 < log_lambda_min < 5:
    return 0.0

  return -np.inf


def run_mcmc(df_msf_bin,optimal_params,correction_integ_obj,chain_length):

  def log_probability(theta_now):

    lp = log_prior(theta_now)

    if not np.isfinite(lp):
      return -np.inf

    if np.isnan(lp):
      return -np.inf

    if np.isnan(ln_p_sch_mcmc(df_msf_bin,theta_now,correction_integ_obj)):
      return -np.inf
    
    return lp + ln_p_sch_mcmc(df_msf_bin,theta_now,correction_integ_obj)


  pos =  optimal_params + 1e-4 * np.random.randn(32, 3)
  nwalkers, ndim = pos.shape

  sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability)
  sampler.run_mcmc(pos,chain_length, progress=True)

  flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

  print(flat_samples.shape)

  if chain_length > 5000:
    tau = sampler.get_autocorr_time()
    print(tau)

  return flat_samples



def Fagn_mstar_model(log_M, a3, a2, a1, a0):
  
  F = a3*(log_M -10.5)**3 + a2*(log_M - 10.5)**2 + a1*(log_M - 10.5) + a0
  
  return F


def plot_manga_constraint():

  a3_mean = 0.09
  a2_mean = -0.81
  a1_mean = 2.43 
  a0_mean = -2.75

  mu = np.array([a3_mean, a2_mean, a1_mean, a0_mean])

  cov_mat = np.array([[0.0002, -0.0014, 0.0029, -0.0015],
                      [-0.0014, 0.0106, -0.0223, 0.012],
                      [0.003, -0.0223, 0.0486, -0.0276],
                      [-0.0015, 0.012, -0.0276, 0.0175]])
  

  m = 100
  n = 10000
  coeffs_array = np.random.multivariate_normal(mu, cov= cov_mat, size = n)

  log_Ms = np.linspace(10,12,m)
  log_Fs_all = []

  for i in coeffs_array:
      
      a3,a2,a1,a0 = i

      Fs = Fagn_mstar_model(log_Ms,a3,a2,a1,a0)
      log_Fs_all.append(Fs)
  
  log_Fs_all = np.array(log_Fs_all)
  log_Fs_ul = np.percentile(log_Fs_all, 84, axis = 0)
  log_Fs_ll = np.percentile(log_Fs_all, 16, axis = 0)
  log_Fs = np.percentile(log_Fs_all, 50, axis = 0)


  plt.plot(log_Ms, log_Fs, label = r'$M_\star Model$', linewidth = 2.5, c = 'k')
  plt.fill_between(log_Ms, log_Fs_ll, log_Fs_ul, color = 'tab:gray', alpha = 0.5)



def get_Fagn_mbins():
  log_lambda_c = -3
  nbins = 5
  path_inputs_sf = "../AGN_FIRST_files/results/MLE/mass_ssfr_bins/mle_sf"
  path_inputs_q = '../AGN_FIRST_files/results/MLE/mass_ssfr_bins/mle_q'

  inputs_sf = []
  inputs_q = []

  path_mcmcs_sf = "../AGN_FIRST_files/results/MCMC/mass_bins/mcmcsf"
  path_mcmcs_q = "../AGN_FIRST_files/results/MCMC/mass_bins/mcmcq"

  mcmcs_sf = []
  mcmcs_q = []

  for i in range(nbins):
    
    inputs_sf.append(np.load(path_inputs_sf + str(i+1) + ".npz"))
    inputs_q.append(np.load(path_inputs_q + str(i+1) + ".npz"))  

    mcmcs_sf.append(np.load(path_mcmcs_sf + str(i+1) +".npz")['flat_samples'])
    mcmcs_q.append(np.load(path_mcmcs_q + str(i+1) +".npz")['flat_samples'])

  Ms_sf = []
  log_Fracs_sf = []
  log_ebars_sf_lower = []
  log_ebars_sf_upper = []

  Ms_q = []
  log_Fracs_q = []
  log_ebars_q_lower = []
  log_ebars_q_upper = []

  for i in range(nbins):

    Msf = inputs_sf[i]['M']
    alpha_0_sf = inputs_sf[i]['alpha_0']+ 1e-8
    lambda_star_sf = 10**inputs_sf[i]['lambda_star']
    log_lambda_min_sf = inputs_sf[i]['log_lambda_min']

    mcmc_samples = mcmcs_sf[i]

    model_parms = [Msf, alpha_0_sf, lambda_star_sf, log_lambda_min_sf]
    [Msf, log_Fsf, log_ebarssf] = get_absFrac_withebars_mbins(model_parms, mcmc_samples, log_lambda_c)

    Ms_sf.append(Msf)
    log_Fracs_sf.append(log_Fsf)
    log_ebars_sf_lower.append(log_ebarssf[0][0])
    log_ebars_sf_upper.append(log_ebarssf[1][0])

    Mq = inputs_q[i]['M']
    alpha_0_q = inputs_q[i]['alpha_0']
    lambda_star_q = 10**inputs_q[i]['lambda_star']
    log_lambda_min_q = inputs_q[i]['log_lambda_min']

    mcmc_samples = mcmcs_q[i]

    model_parms = [Mq, alpha_0_q, lambda_star_q, log_lambda_min_q]
    [Mq, log_Fq, log_ebarsq] = get_absFrac_withebars_mbins(model_parms, mcmc_samples, log_lambda_c)

    Ms_q.append(Mq)
    log_Fracs_q.append(log_Fq)
    log_ebars_q_lower.append(log_ebarsq[0][0])
    log_ebars_q_upper.append(log_ebarsq[1][0])


  log_ebars_sf_lower = np.array(log_ebars_sf_lower)
  log_ebars_sf_upper = np.array(log_ebars_sf_upper)
  log_ebars_sf = np.array([log_ebars_sf_lower, log_ebars_sf_upper])

  log_ebars_q_lower = np.array(log_ebars_q_lower)
  log_ebars_q_upper = np.array(log_ebars_q_upper)
  log_ebars_q = np.array([log_ebars_q_lower, log_ebars_q_upper])

  return [log_Fracs_sf, log_Fracs_q,log_ebars_sf, log_ebars_q]