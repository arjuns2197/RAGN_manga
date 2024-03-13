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
f = 30
near_seps = []

for i in range(10220):

    if agn.get_nearest_nvss_sep(df_matched,i) != None:
        near_seps.append(agn.get_nearest_nvss_sep(df_matched,i))


hist, bin_edges = np.histogram(near_seps)

plt.plot(bin_edges[:-1],hist, linewidth = 4)
plt.xlabel(r"$\rm Nearest~ Offset/~ arcsec$", fontsize = f)
plt.ylabel(r"$\rm Number~ of~ Sources$", fontsize = f)
plt.xticks([0,50,100,150], fontsize = f)
plt.yticks([400,800,1200], fontsize = f)
plt.show()

