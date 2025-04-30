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
f = 45

df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_new_ERs.pkl")
correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]

print(df_manga[''][136])

"""
alpha_0_sf = inputs['alpha_0_sf']
beta_sf = inputs['beta_sf']
lambda_star_sf = inputs['lambda_star_sf']
log_lambda_min_sf = inputs['log_lambda_min_sf']

alpha_0_q = inputs['alpha_0_q']
beta_q = inputs['beta_q']
lambda_star_q = inputs['lambda_star_q']
log_lambda_min_q = inputs['log_lambda_min_q']


log_ms = np.linspace(10, 12 ,100)
Fagn_sf = []
Fagn_q = []

for i in range(len(log_ms)):
  Fagn_sf.append(agn.prob_agnness(log_ms[i],alpha_0_sf,beta_sf,lambda_star_sf,log_lambda_min_sf,-1.5,correc_obj))
  Fagn_q.append(agn.prob_agnness(log_ms[i],alpha_0_q,beta_q,lambda_star_q,log_lambda_min_q,-1.5,correc_obj))


plt.plot(log_ms, Fagn_sf, label = 'SF', c = 'b')
plt.plot(log_ms, Fagn_q, label = 'Q',c = 'r')
plt.ylabel(r"$F_{\rm AGN}$", fontsize = 20)
plt.xlabel(r"$\log(M_\star)$", fontsize = 20)
plt.legend()
plt.show()
"""

def cubic(log_m, a,b,c,d):
    
    log_F = a*(log_m -10.5)**3 + b*(log_m-10.5)**2 + c*(log_m-10.5) + d

    return log_F

abs_fracc = np.load("../AGN_FIRST_files/results/frac/absF_lamc_q_1e-3_correct_ebars.npz",allow_pickle= True)
[F_mean, log_M_mean, ebars] = abs_fracc['abs_fracs']

popt_cubic, pcov = curve_fit(cubic,log_M_mean,np.log10(F_mean))
p,q,r,s = popt_cubic

print(p,q,r,s)
y_predict_cube = cubic(log_M_mean,p,q,r,s)



def plot_frac(property):


    if property == 'log_Mass':
      [frac_prop_mc, prop_mc, errorbar_prop] = abs_fracc['abs_fracs']


    #plt.errorbar(prop_mc,frac_prop_mc,yerr = errorbar_prop,c = 'k',fmt = 'H',ms = 1.5*15,capsize = 1.2*10)  
    #plt.plot(log_M_mean,10**y_predict_cube,linewidth = 6, c = 'k', linestyle = 'dashed',label = "Cubic Fit")
    plt.plot(prop_mc,frac_prop_mc, c = 'r', linewidth = 4, label = r"$\rm Q$")
    plt.fill_between(prop_mc, frac_prop_mc - errorbar_prop[0,:], frac_prop_mc + errorbar_prop[1,:], color = '0.5', alpha = 0.5)
    plt.yscale('log')
    plt.ylim(10**(-5.5),0)
    plt.yticks([10**(-4.5),1e-3,10**(-1.5),1], labels = [-4.5,-3,-1.5,0])
    plt.minorticks_off()
    plt.tick_params(axis ='y',labelsize =f)
    plt.ylabel(r"$\log_{\rm 10} ~\rm F_{\rm AGN}$", fontsize = 1.2*f)
    plt.legend(prop = {'size':1.2*f})


f = 30


plot_frac('log_Mass')
plt.title(r"$M_\star \rm Model$", fontsize =30)
plt.xlabel(r"$\log_{\rm 10}\left(\rm M_\star/ \rm ~M_\odot \right)$", fontsize = f)
plt.tick_params(axis ='x',labelsize =f)
plt.xlim(9.5,12.5)
plt.xticks([10,11,12])
plt.legend(prop = {'size' : f})


abs_fracc = np.load("../AGN_FIRST_files/results/frac/absF_lamc_sf_1e-3_correct_ebars.npz",allow_pickle= True)
[F_mean, log_M_mean, ebars] = abs_fracc['abs_fracs']

popt_cubic, pcov = curve_fit(cubic,log_M_mean,np.log10(F_mean))
p,q,r,s = popt_cubic

print(p,q,r,s)
y_predict_cube = cubic(log_M_mean,p,q,r,s)



def plot_frac(property):


    if property == 'log_Mass':
      [frac_prop_mc, prop_mc, errorbar_prop] = abs_fracc['abs_fracs']


    #plt.errorbar(prop_mc,frac_prop_mc,yerr = errorbar_prop,c = 'k',fmt = 'H',ms = 1.5*15,capsize = 1.2*10)  
    #plt.plot(log_M_mean,10**y_predict_cube,linewidth = 6, c = 'k', linestyle = 'dashed',label = "Cubic Fit")
    plt.plot(prop_mc,frac_prop_mc, c = 'b', linewidth = 4, label = r"$\rm SF$")
    plt.fill_between(prop_mc, frac_prop_mc - errorbar_prop[0,:], frac_prop_mc + errorbar_prop[1,:], color = '0.5', alpha = 0.5)
    plt.yscale('log')
    plt.ylim(10**(-5.5),0)
    plt.yticks([10**(-4.5),1e-3,10**(-1.5),1], labels = [-4.5,-3,-1.5,0])
    plt.minorticks_off()
    plt.tick_params(axis ='y',labelsize =f)
    plt.ylabel(r"$\log_{\rm 10} ~\rm F_{\rm AGN}$", fontsize = 1.2*f)
    plt.legend(prop = {'size':1.2*f})


f = 30


plot_frac('log_Mass')
plt.title(r"$M_\star \rm Model$", fontsize =30)
plt.xlabel(r"$\log_{\rm 10}\left(\rm M_\star/ \rm ~M_\odot \right)$", fontsize = f)
plt.tick_params(axis ='x',labelsize =f)
plt.xlim(9.5,12.5)
plt.xticks([10,11,12])
plt.legend(prop = {'size' : f})



abs_fracc = np.load("../AGN_FIRST_files/results/frac/absF_lamc_1e-3_correct_ebars.npz",allow_pickle= True)
[F_mean, log_M_mean, ebars] = abs_fracc['abs_fracs']

popt_cubic, pcov = curve_fit(cubic,log_M_mean,np.log10(F_mean))
p,q,r,s = popt_cubic

print(p,q,r,s)
y_predict_cube = cubic(log_M_mean,p,q,r,s)



def plot_frac(property):


    if property == 'log_Mass':
      [frac_prop_mc, prop_mc, errorbar_prop] = abs_fracc['abs_fracs']


    #plt.errorbar(prop_mc,frac_prop_mc,yerr = errorbar_prop,c = 'k',fmt = 'H',ms = 1.5*15,capsize = 1.2*10)  
    #plt.plot(log_M_mean,10**y_predict_cube,linewidth = 6, c = 'k', linestyle = 'dashed',label = "Cubic Fit")
    plt.plot(prop_mc,frac_prop_mc, c = '0.4', linewidth = 4, label = r"All")
    plt.fill_between(prop_mc, frac_prop_mc - errorbar_prop[0,:], frac_prop_mc + errorbar_prop[1,:], color = '0.5', alpha = 0.5)
    plt.yscale('log')
    plt.ylim(10**(-5.5),0)
    plt.yticks([10**(-4.5),1e-3,10**(-1.5),1], labels = [-4.5,-3,-1.5,0])
    plt.minorticks_off()
    plt.tick_params(axis ='y',labelsize =f)
    plt.ylabel(r"$\log_{\rm 10} ~\rm F_{\rm AGN}$", fontsize = 1.2*f)
    plt.legend(prop = {'size':1.2*f})


f = 30


plot_frac('log_Mass')
plt.title(r"$M_\star \rm Model$", fontsize =30)
plt.xlabel(r"$\log_{\rm 10}\left(\rm M_\star/ \rm ~M_\odot \right)$", fontsize = f)
plt.tick_params(axis ='x',labelsize =f)
plt.xlim(9.5,12.5)
plt.xticks([10,11,12])
plt.legend(prop = {'size' : f})
plt.show()