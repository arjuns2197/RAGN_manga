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



df_matched = pd.read_pickle('../AGN_FIRST_files/data/df_matched_6733.pkl')

ind = []

for i in range(6733):
    if df_matched['Flux'][i] > 500:
        if agn.is_vi_source(df_matched,i) == True:
            ind.append(i)


print(len(ind))

print(df_matched['plateifu'][ind])

for i in ind:
    print(df_matched['RA'][i], df_matched['DEC'][i])

plt.scatter(df_matched['log_Mass'], df_matched['Lr'])
plt.scatter(df_matched['log_Mass'][ind], df_matched["Lr"][ind], c ='k')
plt.show()