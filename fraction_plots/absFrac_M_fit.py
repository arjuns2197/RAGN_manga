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




'''

mcmc_samples = np.load("../AGN_FIRST_files/results/MCMC/mcmc_output.npz")
samples = mcmc_samples['flat_samples']


alpha0s = samples[:,0]
betas = samples[:,1]
lamstars = samples[:,2]
log_mins = samples[:,3]

inds_alps = np.where((alpha0s > 2.23) & (alpha0s < 2.45))
inds_bets = np.where((betas > -6.6) & (betas < -5.71))
inds_lamstars = np.where((lamstars > 0.87) & (lamstars < 1.5))
inds_mins = np.where((log_mins > -4.54) & (lamstars < -4.26))

alpha0s = alpha0s[inds_alps]
betas = betas[inds_alps]
lamstars = lamstars[inds_alps]
log_mins = log_mins[inds_alps]

def line_eqn(log_mass,a,c):

    log_F = a*(log_mass) + c 

    return log_F

exponents = []

for i in range(len(alpha0s)):

    model_parms = [alpha0s[i],betas[i],lamstars[i],log_mins[i]]
    [f_prop,prop_center] = agn.get_abs_agn_frac("log_Mass",df_manga,model_parms,0.01)

    ind_good_frac = np.where(f_prop > 0)

    f_prop = np.log10(f_prop[ind_good_frac])
    prop_center = prop_center[ind_good_frac]

    popt, pcov = curve_fit(line_eqn, prop_center, f_prop)

    a = popt[0]
    c = popt[1]

    print(a,c)

    log_ms = np.linspace(9.5,12,500)
    plt.scatter(prop_center, f_prop)
    plt.plot(log_ms,line_eqn(log_ms,a,c))
    plt.show()    

'''
correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]

df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6731.pkl")
mcmc_samples = np.load("..\mcmc_output.npz")
samples = mcmc_samples['flat_samples']
optimal_parms = np.load('../AGN_FIRST_files/results/MLE/mass_model/mle_mass.npz')

alp_opt = optimal_parms['alpha_0']
bet_opt = optimal_parms['beta']
lam_opt = optimal_parms['lambda_star']
log_min_opt = optimal_parms['log_lambda_min']

alpha0s = samples[:,0]
betas = samples[:,1]
lamstars = samples[:,2]
log_mins = samples[:,3]

inds_alps = np.where((alpha0s > 2.014) & (alpha0s < 2.194))
inds_bets = np.where((betas > -5.812) & (betas < -5.206))
inds_lamstars = np.where((lamstars > 0.603) & (lamstars < 0.888))
inds_mins = np.where((log_mins > -4.713) & (lamstars < -4.299))

alpha0s = alpha0s[inds_alps]
betas = betas[inds_alps]
lamstars = lamstars[inds_alps]
log_mins = log_mins[inds_alps]



def cubic(log_m, a,b,c,d):
    
    log_F = a*(log_m -10.5)**3 + b*(log_m-10.5)**2 + c*(log_m-10.5) + d

    return log_F

a_array = []
b_array = []
c_array = []
d_array = []

for i in range(len(alpha0s)):

    opt_parms = [alp_opt,bet_opt,lam_opt,log_min_opt]

    [F_mean, log_M_mean] = agn.get_abs_agn_frac(opt_parms,-2,10)

    log_F_mean = np.log10(F_mean)

    popt_mean, pcov = curve_fit(cubic,log_M_mean,log_F_mean)

    print(popt_mean)

    alpha0 = alpha0s[i]
    beta = betas[i]
    lambda_star = lamstars[i]
    log_lambda_min = log_mins[i]

    model_parms = [alpha0,beta,lambda_star,log_lambda_min]

    [F,log_M] = agn.get_abs_agn_frac(model_parms,-2,10)

    log_F = np.log10(F)

    popt, pcov = curve_fit(cubic, log_M, log_F)
    [a,b,c,d] = popt

    a_array.append(a)
    b_array.append(b)
    c_array.append(c)
    d_array.append(d)

a_array = np.array(a_array)
b_array = np.array(b_array)
c_array = np.array(c_array)
d_array = np.array(d_array)

np.savez_compressed("../AGN_FIRST_files/results/frac/abs_F_fit.npz", a_array = a_array,
b_array = b_array, c_array = c_array, d_array = d_array)

    






