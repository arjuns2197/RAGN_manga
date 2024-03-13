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

[seps_mid, seps_far, ind_sources, ind_vi, ind_cands] = agn.get_nvss_3_seps(df_matched)


xticks = [0,50,100,150]
yticks = [50,100,150]
f = 35

plt.scatter(seps_mid,seps_far,c = 'tab:gray',s =120)
plt.scatter(seps_mid[ind_cands],seps_far[ind_cands],c = 'k',s =100, label = 'Candidate NVSS Triple')
plt.scatter(seps_mid[ind_sources],seps_far[ind_sources], facecolors = 'none', edgecolors = 'k', marker = 's', s = 250, label = "Accepted NVSS Triple")
plt.scatter(seps_mid[ind_vi],seps_far[ind_vi], facecolors = 'none', edgecolors = 'k', marker = 'D',
s = 250, label = "Needs VI")
plt.scatter([],[],marker = 'x', label = 'Potential Double', s = 250)
plt.ylim(0,180)
plt.xlim(0,180)
plt.xticks(xticks,fontsize = f)
plt.yticks(yticks, fontsize = f)
plt.xlabel(r"$\rm Second~ Source~ Offset/ ~arcsec$", fontsize = f)
plt.ylabel(r"$\rm Thirdr~ Source~ Offset/~ arcsec$", fontsize = f)
plt.legend(loc = 'lower right', fontsize = f/2)
plt.show()





