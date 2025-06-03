from pdb import line_prefix
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from AGN_FIRST_code import agn_functions as agn

from scipy.special import gammaincc
import astropy
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.table import vstack
import pandas as pd
from scipy import integrate
from scipy import interpolate
import emcee
import time
import corner
log_lambda_max = 2

start = time.time()
bin_num = 1
galaxy_type = 'sf'
correction_integ_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]
chain_length = 10000
df_msf_bin = pd.read_pickle("../AGN_FIRST_files/data/df_M_ssfr_bins/df_" + galaxy_type + str(bin_num) + ".pkl")
inputs = np.load('../AGN_FIRST_files/results/MLE/mass_ssfr_bins/mle_' + galaxy_type + str(bin_num) + '.npz')
alpha_0_opt = inputs['alpha_0']
log_lambda_star_opt = inputs['lambda_star']
log_lambda_min_opt = inputs['log_lambda_min']
optimal_params = [alpha_0_opt,log_lambda_star_opt,log_lambda_min_opt]
log_lambda_c = -3

print(alpha_0_opt, log_lambda_star_opt, log_lambda_min_opt)

# flat_samples = agn.run_mcmc(df_msf_bin,optimal_params,correction_integ_obj,chain_length)
# np.savez_compressed("../AGN_FIRST_files/results/MCMC/mass_bins/mcmc" + galaxy_type + str(bin_num) + ".npz", flat_samples = flat_samples)


flat_samples = np.load("../AGN_FIRST_files/results/MCMC/mass_bins/mcmc"+ galaxy_type + str(bin_num) + ".npz")['flat_samples']
labels = [r'$\alpha$', r'$\log_{\rm 10} \left( \lambda_{\ast} \right)$', r"$\log_{\rm 10} \left( \lambda_{\rm min}\right)$",
r"$\log_{\rm 10} \left(F_{\rm AGN}\right)$"]


Fagns = []

for i in range(len(flat_samples[:,0])):

    alpha_0 = flat_samples[:,0][i]
    lambda_star = 10**flat_samples[:,1][i]
    log_lambda_min = flat_samples[:,2][i]

    F_agn = agn.prob_agnness_unif(alpha_0,lambda_star,log_lambda_min,log_lambda_c,correction_integ_obj)
    Fagns.append(F_agn)


Fagns = np.array(Fagns)
log_Fagns = np.log10(Fagns)

valid_inds = ~np.isnan(log_Fagns)

log_Fagns = log_Fagns[valid_inds].reshape(-1,1)
flat_samples = flat_samples[valid_inds]
flat_samples = np.hstack((flat_samples, log_Fagns))

optimal_Fagn = agn.prob_agnness_unif(alpha_0_opt,10**log_lambda_star_opt,log_lambda_min_opt,log_lambda_c,correction_integ_obj)
optimal_params = np.append(optimal_params, np.log10(optimal_Fagn))

range=[(0.5, 5),(-3,-1),(-8.5, -4.5),(-25,0)]


fig = corner.corner(
    flat_samples,
    labels=labels,
    truths=optimal_params,
    range=range,
    label_kwargs={"fontsize": 14},   # Axis label font size
    tick_kwargs={"labelsize": 12}    # Tick label font size
)

plt.show()