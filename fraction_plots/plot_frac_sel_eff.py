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
f = 30

df_manga = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_new_ERs.pkl")


agn.plot_Fagn_mbins_constraint()
agn.plot_frac_sfq("log_Mass", df_manga)
plt.title("Accounting for Selection Effects is Important!", fontsize =30)
plt.xlabel(r"$\log_{\rm 10}\left(\rm M_\star/ \rm ~M_\odot \right)$", fontsize = f)
plt.tick_params(axis ='x',labelsize = f)
plt.tick_params(axis ='y',labelsize = f)
plt.xlim(10.2,12.2)
plt.xticks([10.6,11,11.4,11.8])
plt.legend(prop = {'size' :0.75*f}, loc = 'lower right')
plt.yscale('log')
plt.minorticks_off()
plt.ylabel(r"$\log_{\rm 10} \left( F_{\rm AGN}\right)$", fontsize = f)
plt.yticks([1e-4,1e-3,1e-2,1e-1,1], labels= [-4,-3,-2,-1,0])
plt.ylim(1e-4,1)
plt.show()