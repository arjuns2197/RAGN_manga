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
[seps_near, seps_mid, ind_sources] = agn.get_first_3_seps(df_matched)


xticks = [0,10,20,30]
yticks = [10,20,30]
f = 35

plt.scatter(seps_near,seps_mid,c = 'tab:gray',s =120, label = "MaNGA galaxy (1 NVSS and 3 FIRST matches)")
plt.scatter(seps_near[ind_sources],seps_mid[ind_sources], facecolors = 'none', edgecolors = 'k', marker = 's', s = 250, label = "Accepted FIRST Triple")
plt.ylim(0,30)
plt.xlim(0,30)
plt.xticks(xticks,fontsize = f)
plt.yticks(yticks, fontsize = f)
plt.xlabel(r"$\rm First~ Source~ Offset/ ~arcsec$", fontsize = f)
plt.ylabel(r"$\rm Secondr~ Source~ Offset/~ arcsec$", fontsize = f)
plt.legend(loc = 'lower right', fontsize = f/2)
plt.show()