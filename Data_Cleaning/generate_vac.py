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
from astropy.table import Table
from astropy.table import QTable


def get_nvss_flux_error(df_matched,index):
  if agn.get_radio_class(df_matched,index) != 0:

    if type(agn.get_radio_class(df_matched,index)) != str:

      flux_array = np.array(df_matched['Flux NVSS(3m)'][index])

      flux = flux_array.sum()
      
      if flux > 2.5:

        flux_error_array = np.array(df_matched['Flux Error NVSS(3m)'][index])
        flux_error = ((flux_error_array**2).sum())**0.5

        return flux_error

      else:
        return -999


    elif type(agn.get_radio_class(df_matched,index)) == str:

      if agn.is_vi_source(df_matched,index) == True:
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


def get_fint_error(theta_M,theta_m,theta_N,peak, rms_pixel_noise):


  fint = agn.get_integ_nvss_flux(theta_M,theta_m,theta_N,peak)

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


def get_descen_flux_inds(df,index,catalog):

  if catalog == "NVSS":
    if df['No. NVSS matches(3m)'][index] > 1:
      inv_sorted_inds = np.argsort(df['Flux NVSS(3m)'][index])[::-1]

  else:
    if df['No. FIRST matches(30s)'][index] > 1:
      inv_sorted_inds = np.argsort(df['Flux FIRST(30s)'][index])[::-1]
  

  return inv_sorted_inds


def sort_subarrays(df,catalog,array):

  new_array = []
  for index in range(10261):

    if catalog == "NVSS":

      if df['No. NVSS matches(3m)'][index] > 1:
          inv_inds = get_descen_flux_inds(df,index,catalog)
          if type(array[index]) != np.ndarray:
            ar = np.array(array[index])
            new_array.append(ar[inv_inds])
        
          else:
            new_array.append(array[index][inv_inds])
         
      else:
        new_array.append(array[index])


    else:
      if df['No. FIRST matches(30s)'][index] > 1:
        inv_inds = get_descen_flux_inds(df,index,catalog)
        if type(array[index]) != np.ndarray:
          ar = np.array(array[index])
          new_array.append(ar[inv_inds])
        
        else:
          new_array.append(array[index][inv_inds])
       
      else:
        new_array.append(array[index])

  new_array = np.array(new_array, dtype= object)
  
  return new_array


def load_drpall_data():

  path_drpall = "../AGN_FIRST_files/data/drpall-v3_1_1.fits"
  data_drpall = fits.open(path_drpall)[1].data

  path_pipe3d = "../AGN_FIRST_files/data/MaNGA_catalog.fits"
  data_pipe3d_all = fits.open(path_pipe3d)[1].data

  print("Initial length:", len(data_drpall))

  mngtarg3 = data_drpall['MNGTARG3']
  
  inds_bad_objs = []

  for i in range(11273):

    if np.log2(mngtarg3[i]) in [8,9,18,19,20,21,22,27]:
      
      inds_bad_objs.append(i)


  data_drpall = np.delete(data_drpall, inds_bad_objs)

  print("Final length: ", len(data_drpall))
  
  plateifu_pipe = data_pipe3d_all['plateifu']
  plateifu_drp = data_drpall['PLATEIFU']

  common_pifu, inds_pipe, inds_drp = np.intersect1d(plateifu_pipe,
  plateifu_drp,return_indices = True)

  data_drpall = Table(data_drpall)
  
  data_drpall['log_sfr_ssp'] = -999*np.ones(len(data_drpall['objra']))
  data_drpall['log_sfr_ssp'][inds_drp] = data_pipe3d_all['log_sfr_ssp'][inds_pipe]

  data_drpall['log_mass'] = -999*np.ones(len(data_drpall['objra']))
  data_drpall['log_mass'][inds_drp] = data_pipe3d_all['log_mass'][inds_pipe]

  ssfr = np.log10(10**data_drpall['log_sfr_ssp'][inds_drp]/10**data_drpall['log_mass'][inds_drp])

  data_drpall['log_ssfr'] = -999*np.ones(len(data_drpall['objra']))
  data_drpall['log_ssfr'][inds_drp] = ssfr

  path_dap = "../AGN_FIRST_files/data/dapall-v3_1_1-3.1.0.fits"
  data_dap = fits.open(path_dap)[1].data

  plateifu_dap = data_dap['PLATEIFU']
  
  common_pifu1, inds_dap, inds_drp = np.intersect1d(plateifu_dap,
  plateifu_drp,return_indices = True)

  data_drpall['log_stellar_sigma_1re'] = -999*np.ones(len(data_drpall['objra']))
  data_drpall['log_stellar_sigma_1re'][inds_drp] = np.log10(data_dap['STELLAR_SIGMA_1RE'][inds_dap])

  return data_drpall




def load_df_vac():

  # load data
  data_drpall = load_drpall_data()
  path_pipe3d = "../AGN_FIRST_files/data/MaNGA_catalog.fits"
  data_pipe3d_all = fits.open(path_pipe3d)[1].data

  #url_nvss = "https://urldefense.proofpoint.com/v2/url?u=https-3A__www.dropbox.com_s_lyuy4lp2lzi5p4d_CATALOG.FIT-3Fdl-3D0&d=DwMFaQ&c=slrrB7dE8n7gBJbeO0g-IQ&r=J9TPhCTsXLU5rvH37Iir6w&m=iqahptD4Y5INWgEYix4rQqPS-qYAZmwpc7Kj0bS2Que1p58myO0ZPWeMXhMgupDZ&s=TbiwgLzBNf8Rusy5ojMbZ8Gjhsc7DWPc0gOhJZvJasw&e="
  path_nvss = '../AGN_FIRST_files/data/nvss_full_catalog.FIT' 
  data_nvss = fits.open(path_nvss,memmap=True)[1].data
  fints = get_integ_nvss_flux(data_nvss['MAJOR AX'], data_nvss['MINOR AX'], 45, data_nvss['PEAK INT'])
  data_nvss = Table(data_nvss)
  data_nvss['FINT'] = fints

  fints_error = get_fint_error(data_nvss['MAJOR AX'],data_nvss['MINOR AX'],45,data_nvss['PEAK INT'],data_nvss['I RMS'])
  data_nvss['FINT ERROR'] = fints_error

  print(data_nvss['PEAK INT'])

  #url_first = 'https://drive.google.com/uc?export=download&id=1itAIFo_i30j9kFxOGxIPDuxuvIJXyUNm'
  path_first = '../AGN_FIRST_files/data/FIRST.fits' 
  data_first = fits.open(path_first,memmap=True)[1].data

  c_nvss = SkyCoord(ra=data_nvss['RA(2000)']*u.degree, dec= data_nvss['DEC(2000)']*u.degree,
  frame= 'icrs', unit='deg')
  c_drpall = SkyCoord(ra=data_drpall['objra']*u.degree, dec= data_drpall['objdec']*u.degree,
  frame= 'icrs', unit='deg') 
  c_first = SkyCoord(ra=data_first['RA']*u.degree, dec= data_first['DEC']*u.degree,
  frame= 'icrs', unit='deg')

  #Match catalogs upto 3 arcmins
  idx_nvss, idx_pipe3d,d2d, d3d = c_drpall.search_around_sky(c_nvss, 3*u.arcmin)
  unique, counts = np.unique(idx_pipe3d, return_counts=True)  

  num_nvss_matches3m = np.zeros(10261)
  num_nvss_matches3m[unique] = counts


  nvss_match_inds3m = agn.get_matching_inds(10261,idx_pipe3d,idx_nvss)

  # FIRST 30 arcsec

  idx_first_30s, idx_pipe3d_first_30s,d2d_first_30s, d3d_first_30s = c_drpall.search_around_sky(c_first,
  30*u.arcsec)
  unique_first_30s, counts_first_30s = np.unique(idx_pipe3d_first_30s,return_counts=True)  
  num_first_matches30s = np.zeros(10261)
  num_first_matches30s[unique_first_30s] = counts_first_30s

  first_match_inds30s = np.zeros(10261, dtype = 'object')
  first_match_inds30s = agn.get_matching_inds(10261,idx_pipe3d_first_30s,
  idx_first_30s)


  # Get RA and DEC of 3 arcmin matches


  ra_first30s = agn.get_property_at_indices(data_first,first_match_inds30s,'RA')
  dec_first30s = agn.get_property_at_indices(data_first,first_match_inds30s,'DEC')

  ra_nvss3m = agn.get_property_at_indices(data_nvss,nvss_match_inds3m,'RA(2000)')
  dec_nvss3m = agn.get_property_at_indices(data_nvss,nvss_match_inds3m,'DEC(2000)')

  flux_first30s = agn.get_property_at_indices(data_first,first_match_inds30s,'FINT')
  flux_nvss3m = agn.get_property_at_indices(data_nvss,nvss_match_inds3m,'FINT')
  flux_nvss3m_error = agn.get_property_at_indices(data_nvss,nvss_match_inds3m,'FINT ERROR')

  maj_ax_first30s = agn.get_property_at_indices(data_first,first_match_inds30s,'MAJOR')
  pos_ang_first30s = agn.get_property_at_indices(data_first,first_match_inds30s,'POSANG')

  z = data_drpall['z']
  ra = data_drpall['objra']
  dec = data_drpall['objdec']
  ra_first = data_first['RA']
  dec_first = data_first['DEC']

  mangaid = data_drpall['mangaid']
  plateifu = data_drpall['plateifu']

  [inds_manga, inds_first] = agn.match_manga_first(ra,dec,ra_first,dec_first,3)

  has_3arcsec_first_match = np.zeros(len(ra))
  has_3arcsec_first_match[inds_manga] = np.ones(len(inds_manga))
  galaxies_drpall = {
  
  "No. NVSS matches(3m)" : num_nvss_matches3m,
  "Indices NVSS matches(3m)" : nvss_match_inds3m,
  "Flux NVSS(3m)":flux_nvss3m,
  "Flux Error NVSS(3m)" : flux_nvss3m_error,
  "RA NVSS(3m)":ra_nvss3m,
  "DEC NVSS(3m)":dec_nvss3m,

  "No. FIRST matches(30s)": num_first_matches30s,
  "Indices FIRST matches(30s)": first_match_inds30s,
  "RA FIRST(30s)":ra_first30s,
  "DEC FIRST(30s)":dec_first30s,
  "Flux FIRST(30s)":flux_first30s,
  "Maj FIRST(30s)": maj_ax_first30s,
  "PA FIRST(30s)":pos_ang_first30s,

  "RA":ra,
  "DEC":dec,
  "3 arcsec FIRST match": has_3arcsec_first_match,
  "nsa_redshift":z,
  "mangaid": mangaid,
  "plateifu":plateifu}


  df_drpall = pd.DataFrame(galaxies_drpall)

  for i in range(10261):
    agn.delete_unrelated_flux_2_1(df_drpall,i)

  for i in range(10261):
    agn.delete_unrelated_flux_3_1(df_drpall,i)
    
  for i in range(10261):
    agn.delete_unrelated_nvss_flux_vi(df_drpall,i)

  radio_class = []

  for i in range(10261):
    if type(agn.get_radio_class(df_drpall,i)) == str:

      if agn.is_vi_source(df_drpall, i) == True:
        radio_class.append(agn.classify_vi_source(df_drpall,i))
      
      else:
        radio_class.append(0)
    
    else:
      radio_class.append(agn.get_radio_class(df_drpall,i))
    
  radio_class = np.array(radio_class, dtype = object)
  df_drpall['Radio Class'] = radio_class


  flux = []

  for i in range(10261):
    flux.append(agn.get_nvss_flux(df_drpall,i))

  flux = np.array(flux)
  df_drpall['fint'] = flux

  flux_error = []

  for i in range(10261):
    flux_error.append(get_nvss_flux_error(df_drpall,i))

  flux_error = np.array(flux_error)
  df_drpall['fint_error'] = flux_error

  plateifu_pipe = data_pipe3d_all['plateifu']
  plateifu_drp = data_drpall['plateifu']

  common_pifu, inds_pipe, inds_drp = np.intersect1d(plateifu_pipe,
  plateifu_drp,return_indices = True)

  sfr_pipe = data_drpall['log_sfr_ssp']
  df_drpall['log_SFR_ssp'] = sfr_pipe

  sigma_pipe = data_drpall['log_stellar_sigma_1re']
  df_drpall['STELLAR_SIGMA_1RE'] = sigma_pipe

  ssfr_pipe = data_drpall['log_ssfr']
  df_drpall['ssfr'] = ssfr_pipe

  stel_mass_pipe = data_drpall['log_mass']
  df_drpall['log_Mass'] = stel_mass_pipe

  inds_radio_dets = np.where(flux > 0)

  Lr = -999*np.ones(10261)
  Lr[inds_radio_dets] = agn.get_radio(flux[inds_radio_dets],z[inds_radio_dets])
  df_drpall['Lr'] = Lr

  log_ER = -999*np.ones(10261)
  log_ER[inds_radio_dets] = np.log10(agn.get_ER(sigma_pipe[inds_radio_dets],Lr[inds_radio_dets]))
  df_drpall['ER'] = log_ER

  is_agn = np.zeros(10261, dtype = bool)
  is_agn[inds_drp] = agn.vector_is_agn(sfr_pipe[inds_drp],Lr[inds_drp])
  df_drpall['is_agn'] = is_agn

  plt.scatter(sfr_pipe[inds_radio_dets],Lr[inds_radio_dets])
  plt.show()
  
  agn_flux_detectable = np.zeros(10261, dtype = bool)
  agn_flux_detectable[inds_drp] = agn.vector_flux_detected(sfr_pipe[inds_drp],Lr[inds_drp],z[inds_drp],check_agn = True)
  df_drpall['agn flux detectable'] = agn_flux_detectable

  data_pipe_erd = agn.load_pipe_data()
  plateifu_pipe_erd = data_pipe_erd['plateifu'] 

  common_pifu_erd, inds_pipe_erd, inds_drp_erd = np.intersect1d(plateifu_pipe_erd,
  plateifu_drp,return_indices = True)

  Lr_lim_flux = -999*np.ones(10261)
  Lr_lim_flux[inds_drp_erd] = agn.get_L_lim(z[inds_drp_erd])
  df_drpall['log_Lr_limit'] = Lr_lim_flux

  Lr_agn_thresh = -999*np.ones(10261)
  Lr_agn_thresh[inds_drp_erd] = agn.v_get_AGN_lim(sfr_pipe[inds_drp_erd],z[inds_drp_erd])
  df_drpall['log_Lr_agn_limit'] = Lr_agn_thresh

  in_erd_analysis = np.zeros(10261, dtype = bool)
  in_erd_analysis[inds_drp_erd] = np.ones(len(in_erd_analysis[inds_drp_erd]), dtype = bool)


  # Sorting subarrays in descending order of flux densities 

  flux_first30s = sort_subarrays(df_drpall,"FIRST",df_drpall['Flux FIRST(30s)'])
  flux_nvss3m = sort_subarrays(df_drpall,"NVSS",df_drpall['Flux NVSS(3m)'])

  flux_first30s_flat = flatten_array(flux_first30s, np.float32)
  flux_nvss3m_flat = flatten_array(flux_nvss3m, np.float32)

  num_nvss_matches3m = df_drpall['No. NVSS matches(3m)']

  nvss_match_inds3m = sort_subarrays(df_drpall,"NVSS",df_drpall['Indices NVSS matches(3m)'])
  nvss_match_inds3m_flat = flatten_array(nvss_match_inds3m,int)


  num_first_matches30s = df_drpall['No. FIRST matches(30s)']
  first_match_inds30s = sort_subarrays(df_drpall,"FIRST", df_drpall['Indices FIRST matches(30s)'])
  first_match_inds30s_flat = flatten_array(first_match_inds30s,int)

  # Get RA and DEC of 3 arcmin matches

  ra_first30s = sort_subarrays(df_drpall,"FIRST",df_drpall['RA FIRST(30s)'])
  dec_first30s = sort_subarrays(df_drpall, "FIRST", df_drpall['DEC FIRST(30s)'])

  ra_first30s_flat = flatten_array(ra_first30s, np.float64)
  dec_first30s_flat = flatten_array(dec_first30s, np.float64)

  ra_nvss3m = sort_subarrays(df_drpall,"NVSS",df_drpall['RA NVSS(3m)'])
  dec_nvss3m = sort_subarrays(df_drpall, "NVSS", df_drpall['DEC NVSS(3m)'])

  ra_nvss3m_flat = flatten_array(ra_nvss3m, np.float64)
  dec_nvss3m_flat = flatten_array(dec_nvss3m, np.float64)

  flux_nvss3m_error = df_drpall["Flux Error NVSS(3m)"]
  flux_nvss3m_error = flatten_array(flux_nvss3m_error, np.float64)

  flux_with_dummy_value = []
  for i in flux:
    if i > 0:
      flux_with_dummy_value.append(i)
    
    else:
      flux_with_dummy_value.append(-999)

  flux_with_dummy_value = np.array(flux_with_dummy_value)

  flux_error_with_dummy_value = []
  for i in flux_error:
    if i > 0:
      flux_error_with_dummy_value.append(i)
    
    else:
      flux_error_with_dummy_value.append(-999)



  c1 = fits.Column(name='mangaid', array= mangaid, format = '32A')
  c2 = fits.Column(name='plateifu', array= plateifu, format = '32A')
  c3 = fits.Column(name='ra', array= ra, format = 'D')
  c4 = fits.Column(name='dec', array= dec, format = 'D')
  c5 = fits.Column(name='fint', array= flux_with_dummy_value, format = 'D')
  c6 = fits.Column(name='log_lr', array= Lr, format = 'D')
  c7 = fits.Column(name='is_detected_agn', array= is_agn, format = 'L')
  c8 = fits.Column(name='radio_class', array= radio_class, format = 'I')
  c9 = fits.Column(name='num_nvss_3m', array= num_nvss_matches3m, format = 'I')
  c10 = fits.Column(name='ras_nvss_3m', array= ra_nvss3m_flat, format = '6D')
  c11 = fits.Column(name='decs_nvss_3m', array= dec_nvss3m_flat, format = '6D')
  c12 = fits.Column(name='inds_nvss_3m', array= nvss_match_inds3m_flat, format = '6K')
  c13 = fits.Column(name='fluxes_nvss_3m', array= flux_nvss3m_flat, format = '6D')
  c14 = fits.Column(name='num_first_30s', array= num_first_matches30s, format = 'I')
  c15 = fits.Column(name='ras_first_30s', array= ra_first30s_flat, format = '5D')
  c16 = fits.Column(name='decs_first_30s', array= dec_first30s_flat, format = '5D')
  c17 = fits.Column(name='inds_first_30s', array= first_match_inds30s_flat, format = '5K')
  c18 = fits.Column(name='fluxes_first_30s', array= flux_first30s_flat, format = '5D')
  c19 = fits.Column(name='log_sfr_ssp', array= sfr_pipe, format = 'D')
  c20 = fits.Column(name='log_ssfr', array= ssfr_pipe, format = 'D')
  c21 = fits.Column(name='log_mass', array= stel_mass_pipe, format = 'D')
  c22 = fits.Column(name='log_stellar_sigma_1re', array= sigma_pipe, format = 'D')
  c23 = fits.Column(name='nsa_redshift', array= z, format = 'D')
  c24 = fits.Column(name='log_lr_limit', array= Lr_lim_flux, format = 'D')
  c25 = fits.Column(name='log_lr_agn_thresh', array= Lr_agn_thresh, format = 'D')
  c26 = fits.Column(name='in_erd_analysis', array= in_erd_analysis, format = 'L')
  c27 = fits.Column(name='flux_errors_nvss_3m', array= flux_nvss3m_error, format = '6D')
  c28 = fits.Column(name='fint_error', array= flux_error_with_dummy_value, format = 'D')
  c29 = fits.Column(name='log_er', array= log_ER, format = 'D')
                    
  t = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,
  c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29])

  t.writeto('../AGN_FIRST_files/results/VAC/VAC.fits', overwrite = True)

  return df_drpall



df_1 = load_df_vac()
df_1.to_pickle("../AGN_FIRST_files/results/VAC/df_vac.pkl")

df = fits.open('../AGN_FIRST_files/results/VAC/VAC.fits')[1].data



inds_erd = df['in_erd_analysis']

f = df['fint'][inds_erd]
f_err = df['fint_error'][inds_erd]

print(df['Flux FIRST(30s)'][inds_erd][-2])
plt.scatter(f,f_err)
plt.show()

inds = np.where(df['fint'] > 0)[0]


f = df['fint'][inds]
f_error = df['fint_error'][inds]


for i in range(1126):
  if f_error[i]/f[i] < 0.03:
    print(df['num_nvss_3m'][inds[i]])









