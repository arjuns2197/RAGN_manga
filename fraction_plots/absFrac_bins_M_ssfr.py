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
f = 30

log_lambda_c = -3
correc_obj = np.load("../AGN_FIRST_files/data/interp_obj.npy", allow_pickle= True)[()]

[log_Fracs_sf, log_Fracs_q,log_ebars_sf, log_ebars_q] = agn.get_Fagn_mbins()

print(log_ebars_q)

galaxy_types = ['q', 'sf']
bin_nums = np.arange(1,3)

log_Fagn_uls = []

for galaxy_type in galaxy_types:
  for bin_num in bin_nums:

    path = "../AGN_FIRST_files/data/df_M_ssfr_bins/df_" + galaxy_type + str(bin_num) + ".pkl"
    df = pd.read_pickle(path)

    vd = df['STELLAR_SIGMA_1RE'].to_numpy()
    Lr = df['Lr'].to_numpy()
    z = df['nsa_redshift'].to_numpy()
    sfr = df['log_SFR_SF'].to_numpy()
    is_agn = df['agn flux detectable'].to_numpy(dtype = bool)

    lambda_lims = agn.get_ER_lim_agn_cav(vd, z, sfr)
    lambdas = agn.vector_er_pcav_check_agn(Lr, vd, is_agn)

    log_lambda_lims = np.log10(lambda_lims)
    log_lambdas = np.log10(lambdas)

    inds_lt_minus3 = log_lambda_lims < -3
    log_lambda_lims = log_lambda_lims[inds_lt_minus3]

    ul = np.log10(1/len(log_lambda_lims))
    log_Fagn_uls.append(ul) 


Ms = [10.2, 10.6,11, 11.4, 11.8]
agn.plot_manga_constraint()
plt.plot(Ms[1:],log_Fracs_sf[1:], c = 'b', linewidth = 1.5)
plt.errorbar(Ms[1:],log_Fracs_sf[1:],yerr = log_ebars_sf[:, 1:], fmt = 'H',c = 'b', ms = 15, capsize = 17, label = 'SF', elinewidth= 1, alpha = 0.75)
plt.errorbar(Ms[2:],log_Fracs_q[2:],yerr = log_ebars_q[:, 2:], fmt = 'H',c = 'r', ms = 15, capsize = 17, label = 'Q', elinewidth= 1, alpha = 0.75)
plt.plot(Ms[2:], log_Fracs_q[2:], c = 'r', linewidth = 1.5)
plt.ylim(-5,1)
plt.minorticks_off()
plt.tick_params(axis ='y',labelsize =f)
plt.ylabel(r"$\log_{\rm 10} ~\rm F_{\rm AGN}$", fontsize = 1.2*f)
plt.xlabel(r"$\log_{\rm 10}\left(\rm M_\star/ \rm ~M_\odot \right)$", fontsize = f)
plt.tick_params(axis ='x',labelsize =f)
plt.xlim(9.5,12.5)
plt.xticks([10,10.5,11,11.5,12])

#Q1 bin
plt.hlines(log_Fagn_uls[0], (Ms[0]) - 0.05, (Ms[0]) + 0.05, colors='r', linewidth=1)  # thin horizontal line
plt.annotate('', xy=((Ms[0]), log_Fagn_uls[0] - 0.3), xytext=((Ms[0]), log_Fagn_uls[0]),
arrowprops=dict(arrowstyle='-|>', color='r', linewidth=1))

#Q2 bin
plt.hlines(log_Fagn_uls[1], (Ms[1] + 0.01) - 0.05, (Ms[1] + 0.01) + 0.05, colors='r', linewidth=1)  # thin horizontal line
plt.annotate('', xy=((Ms[1] + 0.01), log_Fagn_uls[1] - 0.3), xytext=((Ms[1] + 0.01), log_Fagn_uls[1]),
arrowprops=dict(arrowstyle='-|>', color='r', linewidth=1))

#SF1 bin
plt.hlines(log_Fagn_uls[2], (Ms[0]) - 0.05, (Ms[0]) + 0.05, colors='b', linewidth=1)  # thin horizontal line
plt.annotate('', xy=((Ms[0]), log_Fagn_uls[2] - 0.3), xytext=((Ms[0]), log_Fagn_uls[2]),
arrowprops=dict(arrowstyle='-|>', color='b', linewidth=1))

plt.legend(loc='lower right', prop={'size': 1.2 * f})
plt.show()