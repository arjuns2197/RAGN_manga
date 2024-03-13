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

df_matched = pd.read_pickle("../AGN_FIRST_files/data/df_drpmatched.pkl")


pma_cut =  np.linspace(0,30,1500)
seps_of_pma_cut = agn.v_nvss_single1_cut(pma_cut)

[seps, pma,  inds_sources] = agn.get_source1_seps_and_pma(df_matched)

xticks = [10,20,30]
yticks = [0,10,20,30]
f = 35

plt.scatter(seps,pma,c = 'tab:gray',s =120, label = 'MaNGA galaxy (1 NVSS match)')
plt.plot(seps_of_pma_cut,pma_cut, c = 'k', linewidth = 4, label = " Radio Type 1 Candidate Cut")
plt.scatter(seps[inds_sources],pma[inds_sources], facecolors = 'none', edgecolors = 'k', marker = 's', s = 250, label = "Accepted Type 1 Source")
plt.ylim(0,30)
plt.xlim(0,30)
plt.xticks(xticks,fontsize = f)
plt.yticks(yticks, fontsize = f)
plt.xlabel(r"$\rm Nearest~ Offset/ ~arcsec$", fontsize = f)
plt.ylabel(r"$\rm Projected~ Major~ Axis/~ arcsec$", fontsize = f)
plt.legend(loc = 'upper right', fontsize = f/2)
plt.show()
