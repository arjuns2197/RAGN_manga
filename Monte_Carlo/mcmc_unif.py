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
log_lambda_max = 2

start = time.time()

interp_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]
df_matched = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_new_ERs.pkl")

inputs = np.load('../AGN_FIRST_files/results/MLE/unif_model/mle_unif_final.npz')

alpha_0 = inputs['alpha_0']
beta = inputs['beta']
lambda_star = inputs['lambda_star']
log_lambda_min = inputs['log_lambda_min']

optimal_params = [alpha_0,beta,lambda_star,log_lambda_min]

print(optimal_params)

mass = df_matched['log_Mass']
if type(mass) != np.ndarray:
    mass = mass.to_numpy()

vol_weights = np.ones(len(df_matched['ESWEIGHT']))
if type(vol_weights) != np.ndarray:
    vol_weights = vol_weights.to_numpy()

Lr = df_matched['Lr']
if type(Lr) != np.ndarray:
    Lr = Lr.to_numpy()


sfr = df_matched['log_SFR_SF']
if type(sfr) != np.ndarray:
    sfr = sfr.to_numpy()
  

vd = df_matched['STELLAR_SIGMA_1RE']
if type(vd) != np.ndarray:
    vd = vd.to_numpy()

  
z = df_matched['nsa_redshift']
if type(z) != np.ndarray:
    z = z.to_numpy()

ind_agn = df_matched["agn flux detectable"].to_numpy(dtype= bool)

Lr_agn = Lr[ind_agn]
Lr_non_agn = np.delete(Lr, ind_agn)

sfr_agn = sfr[ind_agn]
sfr_non_agn = np.delete(sfr, ind_agn)

vd_agn = vd[ind_agn]
vd_non_agn = np.delete(vd, ind_agn)

z_agn = z[ind_agn]
z_non_agn = np.delete(z, ind_agn)

mass_agn = mass[ind_agn]
mass_non_agn = np.delete(mass, ind_agn)

ssfr_agn = np.log10(10**sfr_agn/10**mass_agn)
ssfr_non_agn = np.log10(10**sfr_non_agn/10**mass_non_agn)

volw_agn = vol_weights[ind_agn]
volw_non_agn = np.delete(vol_weights, ind_agn)

lambda_lim_non_agn = agn.get_ER_lim_agn_cav(vd_non_agn,z_non_agn,sfr_non_agn)
lambda_agn = df_matched["ER_cav"][ind_agn]


def ln_p_sch(theta,property,correction_integ_obj): # Likelihood Function

    if property == "log_Mass":
        prop_agn = mass_agn
        prop_non_agn = mass_non_agn
    
    if property == "log_SFR_SF":
        prop_agn = sfr_agn
        prop_non_agn = sfr_non_agn

    if property == "STELLAR_SIGMA_1RE":
        prop_agn = vd_agn
        prop_non_agn = vd_non_agn
    
    if property == 'sSFR':
        prop_agn = ssfr_agn
        prop_non_agn = ssfr_non_agn
    
    if property == None:
        prop_agn = np.ones(len(sfr_agn))
        prop_non_agn = np.ones(len(sfr_non_agn))

    alpha_0 = theta[0]
    beta = theta[1]
    lambda_star = theta[2]
    log_lambda_min = theta[3]

    alphas_agn = agn.alpha_prop(alpha_0,prop_agn,beta,property)

    inds_alp_lt_1 = np.where(alphas_agn < 1)
    inds_alp_gt_1 = np.where(alphas_agn > 1)

    norms_agn = np.ones(len(alphas_agn))

    norms_agn[inds_alp_gt_1] = 1/(agn.total_integral(log_lambda_min,log_lambda_max,alphas_agn[inds_alp_gt_1],
    lambda_star,correction_integ_obj)) 
    norms_agn[inds_alp_lt_1] = 1/(agn.gamma_integ(log_lambda_min,log_lambda_max,alphas_agn[inds_alp_lt_1],
    lambda_star)) 

    likelihoods_agn = norms_agn*agn.schechter_prop(lambda_agn,
    prop_agn,alpha_0,beta,lambda_star,property)

    ln_likelihood = np.log(likelihoods_agn).sum()

    alphas_non_agn = agn.alpha_prop(alpha_0,prop_non_agn,beta,property)
    norms_non_agn = np.ones(len(alphas_non_agn))

    inds_alps_lt_1 = np.where(alphas_non_agn < 1)
    inds_alps_gt_1 = np.where(alphas_non_agn > 1)

    norms_non_agn[inds_alps_gt_1] = 1/(agn.total_integral(log_lambda_min,log_lambda_max,alphas_non_agn[inds_alps_gt_1],
    lambda_star,correction_integ_obj)) 
    norms_non_agn[inds_alps_lt_1] = 1/(agn.gamma_integ(log_lambda_min,log_lambda_max,alphas_non_agn[inds_alps_lt_1],
    lambda_star))  

    likelihoods_alps_gt_1 = norms_non_agn[inds_alps_gt_1]*agn.total_integral(log_lambda_min,
    np.log10(lambda_lim_non_agn[inds_alps_gt_1]),alphas_non_agn[inds_alps_gt_1],lambda_star,correction_integ_obj)

    likelihoods_alps_lt_1 = norms_non_agn[inds_alps_lt_1]*agn.gamma_integ(log_lambda_min,
    np.log10(lambda_lim_non_agn[inds_alps_lt_1])
    ,alphas_non_agn[inds_alps_lt_1],lambda_star)

    ln_likelihood = ln_likelihood + np.log(likelihoods_alps_gt_1).sum() + np.log(likelihoods_alps_lt_1).sum()

    print(ln_likelihood)

    return ln_likelihood


def log_prior(theta):
    alpha_0, beta, lambda_star,log_lambda_min = theta

    if 1.2 < alpha_0 < 1.6 and -np.inf < beta < np.inf and 7e-3 < lambda_star < 0.1 and -6.5 < log_lambda_min < -4.5:
        return 0.0

    return -np.inf


def log_probability(theta):
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    if np.isnan(lp):
        return -np.inf

    if np.isnan(ln_p_sch(theta,None, interp_obj)):
        return -np.inf
     
    return lp + ln_p_sch(theta,None, interp_obj)



pos =  optimal_params + 1e-2 * np.random.randn(32, 4)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 5000, progress=True)


fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["alpha_0", "beta", "lambda_star", "log_lambda_min"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

import corner

fig = corner.corner(
    flat_samples, labels=labels, truths= optimal_params
)

plt.show()


from IPython.display import display, Math

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\agnhrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    print(txt)


end = time.time()

print(end/60-start/60)
np.savez_compressed("../AGN_FIRST_files/results/MCMC/mcmc_unif_final.npz", flat_samples = flat_samples)
