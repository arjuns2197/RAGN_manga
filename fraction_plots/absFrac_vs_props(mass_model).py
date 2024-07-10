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
from scipy.special import betainc
from scipy.optimize import curve_fit

correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]
df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_new_ERs.pkl")
mcmc_samples = np.load("../AGN_FIRST_files/results/MCMC/mcmc_mass_final.npz")
samples = mcmc_samples['flat_samples']

alpha0s = samples[:,0]
betas = samples[:,1]
lamstars = samples[:,2]
log_mins = samples[:,3]


inds_alps = np.where((alpha0s > 1.89) & (alpha0s < 2.11))
inds_bets = np.where((betas > -5.30) & (betas < -4.71))
inds_lamstars = np.where((lamstars > 0.01) & (lamstars < 0.02))
inds_mins = np.where((log_mins > -5.95) & (lamstars < -5.38))

alpha0s = alpha0s[inds_alps]
betas = betas[inds_alps]
lamstars = lamstars[inds_alps]
log_mins = log_mins[inds_alps]

inputs = np.load('../AGN_FIRST_files/results/MLE/mass_model/mle_mass_final.npz')

alpha_0 = inputs['alpha_0']
beta = inputs['beta']
lambda_star = inputs['lambda_star']
log_lambda_min = inputs['log_lambda_min']


def get_abs_agn_frac(model_parms,lambda_c): # Input exact column name as in Dataframe (str)
 
  '''Returns list of arrrays containing data-AGN-fraction as a function of required property, 
  values of galaxy the property, model-AGN-fraction as a function of required property
  and the errorbars on the model.

  Inputs column name of property, manga dataframe, model dataframe,'''

  nbins = 100

  [alpha_0,beta,lambda_star,log_lambda_min] = model_parms

  prop_center = []
  f_prop = []

  for i in np.linspace(10,12,nbins):
    f_prop.append(agn.prob_agnness(i,alpha_0,beta,lambda_star,log_lambda_min,lambda_c,correc_obj))
    prop_center.append(i)


  f_prop = np.array(f_prop)
  prop_center = np.array(prop_center)

  
  return [f_prop, prop_center]


def get_absF_with_ebars(model_parms,lambda_c):

  [frac, prop_center] = get_abs_agn_frac(model_parms,lambda_c)
  fracs_all = []

  for i in range(len(betas)):

    fracs_all.append(get_abs_agn_frac(
    [alpha0s[i],betas[i],lamstars[i],log_mins[i]],lambda_c)[0])
  
  fracs_all = np.array(fracs_all)

  lbs = []
  ubs = []

  for i in range(len(frac)):

    fs = fracs_all[:,i]
    errors = agn.get_absF_error(fs,frac[i])

    lbs.append(errors[0])
    ubs.append(errors[1])
  
  lbs = np.array(lbs)
  ubs = np.array(ubs)

  ebars = np.array([lbs,ubs])

  return [frac, prop_center,ebars]


abs_fracs = []
prop_array = [ 'log_SFR_ssp', 'sSFR', 'log_Mass','STELLAR_SIGMA_1RE']
for i in prop_array:
  abs_fracs.append(get_absF_with_ebars(
  [alpha_0,beta,lambda_star,log_lambda_min],-3))

np.savez_compressed("../AGN_FIRST_files/results/frac/absF_lamc_1e-3.npz",abs_fracs = abs_fracs, allow_pickle = True)

abs_fracs_lamc_minus2 = []
prop_array = [ 'log_SFR_ssp', 'sSFR', 'log_Mass','STELLAR_SIGMA_1RE']
for i in prop_array:
  abs_fracs_lamc_minus2.append(get_absF_with_ebars(
  [alpha_0,beta,lambda_star,log_lambda_min],-2))

np.savez_compressed("../AGN_FIRST_files/results/frac/absF_lamc_1e-2.npz",abs_fracs = abs_fracs_lamc_minus2, allow_pickle = True)


def cubic(log_m, a,b,c,d):
    
    log_F = a*(log_m -10.5)**3 + b*(log_m-10.5)**2 + c*(log_m-10.5) + d

    return log_F


optimal_parms = np.load('../AGN_FIRST_files/results/MLE/mass_model/mle_mass_final.npz')

alp_opt = optimal_parms['alpha_0']
bet_opt = optimal_parms['beta']
lam_opt = optimal_parms['lambda_star']
log_min_opt = optimal_parms['log_lambda_min']

opt_parms = [alp_opt,bet_opt,lam_opt,log_min_opt]

[F_mean, log_M_mean] = agn.get_abs_agn_frac(opt_parms,-3,10)
popt_cubic, pcov = curve_fit(cubic,log_M_mean,np.log10(F_mean))
p,q,r,s = popt_cubic
print(p,q,r,s)
y_predict_cube = cubic(log_M_mean,p,q,r,s)

y_predict_old = cubic(log_M_mean,0.07,-0.77,2.44,-2.60)


abs_fracc = np.load("../AGN_FIRST_files/results/frac/absF_lamc_1e-3.npz",allow_pickle= True)
abs_fracc = abs_fracc['abs_fracs']

abs_fracc_minus2 = np.load("../AGN_FIRST_files/results/frac/absF_lamc_1e-2.npz",allow_pickle= True)
abs_fracc_minus2 = abs_fracc_minus2['abs_fracs']

def plot_frac(property):

    if property == 'log_SFR_ssp':
      [frac_prop_mc, prop_mc, errorbar_prop] = abs_fracc[0]
      [frac_prop_mc_minus2, prop_mc_minus2, errorbar_prop_minus2] = abs_fracc_minus2[0]

    if property == 'sSFR':
      [frac_prop_mc, prop_mc, errorbar_prop] = abs_fracc[1]
      [frac_prop_mc_minus2, prop_mc_minus2, errorbar_prop_minus2] = abs_fracc_minus2[1]

    if property == 'log_Mass':
      [frac_prop_mc, prop_mc, errorbar_prop] = abs_fracc[2]
      [frac_prop_mc_minus2, prop_mc_minus2, errorbar_prop_minus2] = abs_fracc_minus2[2]

    if property == 'STELLAR_SIGMA_1RE':
      [frac_prop_mc, prop_mc, errorbar_prop] = abs_fracc[3]
      [frac_prop_mc_minus2, prop_mc_minus2, errorbar_prop_minus2] = abs_fracc_minus2[3]
    


    #plt.errorbar(prop_mc,frac_prop_mc,yerr = errorbar_prop,c = 'k',fmt = 'H',ms = 1.5*15,capsize = 1.2*10)  
    #plt.plot(log_M_mean,10**y_predict_cube,linewidth = 6, c = 'k', linestyle = 'dashed',label = "Cubic Fit")
    plt.plot(log_M_mean,10**y_predict_old,linewidth = 6, c = 'r', linestyle = 'dashed',
             label = "Panessa ERs;" + r"$\log \left(\lambda_c\right) = -2$")
    plt.plot(prop_mc,frac_prop_mc, c = '0.4', linewidth = 4, label = r"$M_\star \rm Model~ Prediction; \log \left(\lambda_c\right) = -3$")
    plt.plot(prop_mc_minus2,frac_prop_mc_minus2, c = 'b', linewidth = 4, label = r"$\log \left(\lambda_c\right) = -2$")
    plt.fill_between(prop_mc, frac_prop_mc - errorbar_prop[0,:], frac_prop_mc + errorbar_prop[1,:], color = '0.5', alpha = 0.5)
    plt.yscale('log')
    plt.ylim(10**(-5.5),0)
    plt.yticks([10**(-4.5),1e-3,10**(-1.5),1], labels = [-4.5,-3,-1.5,0])
    plt.minorticks_off()
    plt.tick_params(axis ='y',labelsize =f)
    plt.ylabel(r"$\log_{\rm 10} ~\rm F_{\rm AGN}$", fontsize = 1.2*f)
    plt.legend(prop = {'size':1.2*f})


f = 30
plt.figure(figsize = (20,9))
plt.suptitle(r"$\rm M_\star$ Model", fontsize = f)

properties = ["log_Mass", "STELLAR_SIGMA_1RE","log_SFR_ssp", "sSFR"]

plt.subplots_adjust(hspace = 0.5, wspace = 0.25)

for i in enumerate(properties):
    plt.subplot(2,2,i[0]+1)

    if i[0] == 0:
        
        plt.xlabel(r"$\log_{\rm 10}\left(\rm M_\star/ \rm ~M_\odot \right)$", fontsize = 1.2*f)
        plt.tick_params(axis ='x',labelsize =f)
        plt.xlim(9.5,12.5)
        plt.xticks([10,11,12])
    
    if i[0] == 1:
        
        plt.xlabel(r"$\log_{\rm 10}\left(\sigma_v/\rm ~km~ \rm s^{\rm -1}\right)$", fontsize = 1.2*f)
        plt.tick_params(axis ='x',labelsize =f)
        plt.xlim(1.8,2.7)
        plt.xticks([2,2.2,2.4,2.6])  
        
    if i[0] == 2:
        
        plt.xlabel(r"$\log_{10}\left(\rm SFR/\rm ~M_\odot ~ \rm yr^{\rm -1}\right)$", fontsize = 1.2*f)
        plt.tick_params(axis ='x',labelsize =f)
        plt.xlim(-2.5,2.4)
        plt.xticks([-1.5,-0.5,0.5,1.5])

    if i[0] == 3:
        
        plt.xlabel(r"$\log_{\rm 10}\left(\rm sSFR/ ~ \rm yr^{\rm -1}\right)$", fontsize = 1.2*f)
        plt.tick_params(axis ='x',labelsize =f)
        plt.xlim(-13.75,-9.5)
        plt.xticks([-13,-12,-11,-10])
         
        
    plot_frac(i[1])


plt.show()




plot_frac('log_Mass')
plt.title(r"$M_\star \rm Model$", fontsize =30)
plt.xlabel(r"$\log_{\rm 10}\left(\rm M_\star/ \rm ~M_\odot \right)$", fontsize = f)
plt.tick_params(axis ='x',labelsize =f)
plt.xlim(9.5,12.5)
plt.xticks([10,11,12])
plt.legend(prop = {'size' : f})
plt.show()