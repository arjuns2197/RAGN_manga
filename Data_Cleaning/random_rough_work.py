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


file_name = '../AGN_FIRST_files/results/VAC/manga_radio_vac_1.0.fits'  # replace with your .fits file
data = fits.open(file_name)[1].data
data = Table(data)
data_head = data[0:20]

df = pd.DataFrame(data_head['mangaid'])

for i in data_head.columns:

    array = np.array(data_head[i])
    df[i] = list(array)
    


df.to_latex('../AGN_FIRST_files/results/VAC/example_table.tex', index = False,longtable= True, float_format="%.2f")

