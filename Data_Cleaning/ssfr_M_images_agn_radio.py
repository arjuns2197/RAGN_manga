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
import imageio.v2 as iio
import requests
from PIL import Image
import matplotlib.gridspec
from math import modf



def point_exists_in_grid(df, M1, M2, ssfr1, ssfr2):

    for i in range(6255):
        if df['log_Mass'][i] > M1 and df['log_Mass'][i] < M2 and df['sSFR'][i] > ssfr1 and df['sSFR'][i] < ssfr2:
            if df['agn flux detectable'][i] == True:
                if df['Radio Class'][i] != 3:
                    if df['Flux FIRST(30s)'][i].sum() > 0.5:

                        return True
        
    else:
        return False


def get_hmsdms(ra, dec):

    c = SkyCoord(ra, dec, frame = 'icrs', unit = 'degree')
    string = c.to_string('hmsdms')
    
    ra, dec = string.split(" ")
    
    hh = ra[0:2]
    mm = ra[3:5]
    ss = ra[6:13]
    
    dd = dec[0:3]
    mm_dec = dec[4:6]
    ss_dec = dec[7:14]

    raa = hh + " " + mm + " " + ss + " "
    decc = dd + " " + mm_dec + ' ' + ss_dec
    
    return raa + decc

def get_random_position(df,M1, M2, ssfr1, ssfr2):

    if point_exists_in_grid(df,M1, M2, ssfr1, ssfr2) == True:

        inds = []

        for i in range(6255):
            if df['log_Mass'][i] > M1 and df['log_Mass'][i] < M2 and df['sSFR'][i] > ssfr1 and df['sSFR'][i] < ssfr2:
                if df['agn flux detectable'][i] == True:
                    if df['No. FIRST matches(30s)'][i] > 1:
                        inds.append(i)
        

        if len(inds) == 0:
            for i in range(6255):
                if df['log_Mass'][i] > M1 and df['log_Mass'][i] < M2 and df['sSFR'][i] > ssfr1 and df['sSFR'][i] < ssfr2:
                    if df['agn flux detectable'][i] == True:
                        if df['No. FIRST matches(30s)'][i] == 1:
                            inds.append(i)


        random_ind = np.random.choice(inds)
        rand_RA = df['RA'][random_ind]
        rand_dec = df['DEC'][random_ind]

        rand_pos = get_hmsdms(rand_RA, rand_dec)
        print(random_ind)

        return rand_pos


f = 30
df_matched = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_Ha.pkl")
inds_agn = df_matched['agn flux detectable'].to_numpy(dtype = bool)

ssfr = df_matched['sSFR']
M = df_matched['log_Mass']
sfr = df_matched['log_SFR_ssp']

print(df_matched['log_Mass'][inds_agn].min())

xmin = 10
xmax = 12
ymin = -15.75
ymax = -9

n = int(10)


figure = plt.figure(figsize = (10,9.8))
ax = figure.add_gridspec(top=0.8, right=0.8).subplots()

ax_histx = ax.inset_axes([0, 1.05, 1, 0.2])
ax_histx.set_xticklabels(labels = [])
ax_histx.set_xticks([])
ax_histx.set_ylabel(r"$\rm N_{\rm AGN}$", fontsize = 0.8*f)
ax_histx.set_yticks([40,80])
ax_histx.set_yticklabels(labels = [40,80], fontsize = 0.8*f)
ax_histx.hist(M[inds_agn], bins = n, color = 'tab:gray', range = (xmin,xmax), alpha = 0.4)
ax_histx.set_xlim(xmin,xmax)

ax_histy = ax.inset_axes([1.05, 0, 0.2, 1])
ax_histy.set_yticklabels(labels = [])
ax_histy.set_yticks([])
ax_histy.set_xlabel(r"$\rm N_{\rm AGN}$", fontsize = 0.8*f)
ax_histy.xaxis.set_label_position('top')
ax_histy.xaxis.tick_top()
ax_histy.hist(ssfr[inds_agn], bins = n, orientation = 'horizontal',color = 'tab:gray', range = (ymin, ymax), alpha = 0.4)
ax_histy.set_xticks([25,50])
ax_histy.set_xticklabels(labels = [25,50], fontsize = 0.8*f, rotation = '-90')
ax_histy.set_ylim(ymin,ymax)


ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xlabel(r"$\log\left(M_\star/M_\odot\right)$", fontsize = f)
ax.set_ylabel(r"$\log\left(\rm sSFR/yr^{-1}\right)$", fontsize = f)
ax.set_xticks([10.3,10.8,11.3,11.8])
ax.set_yticks([-15,-14,-13,-12, -11, -10])
ax.set_xticklabels(labels = [10.3,10.8,11.3,11.8],fontsize = f)
ax.set_yticklabels(labels = [-15,-14,-13,-12, -11, -10],fontsize = f)


for i in range(n):
    for j in range(n):
        M1 = xmin + i*(xmax - xmin)/n
        M2 = xmin + (i+1)*(xmax - xmin)/n
        ssfr1 = ymin + j*(ymax - ymin)/n
        ssfr2 = ymin + (j+1)*(ymax - ymin)/n

        if point_exists_in_grid(df_matched,M1,M2, ssfr1,ssfr2) == True:
            
            inset = ax.inset_axes([i/n, j/n, 1/n, 1/n])
            inset.set_xticks([])
            inset.set_yticks([])

            invalid_file_ignored = 0
            while True:
                try:
                    
                    random_pos = get_random_position(df_matched,M1,M2, ssfr1,ssfr2)

                    payload = {"RA": random_pos,
                    "Equinox": "J2000",
                    "ImageSize":"1.5",
                    "ImageType": "FITS File",
                    "MaxInt": "2000",
                    ".submit":"Extract the Cutout",
                    ".cgifields": "ImageType"}
                    
                    image_path = "../AGN_FIRST_files/data/radio_images/radio_image.fits"
                    r = requests.post('https://third.ucllnl.org/cgi-bin/firstcutout', data = payload)
                    with open(image_path, "wb") as f:
                        f.write(r.content)
            
                    img = -fits.open(image_path)[0].data
                    inset.imshow(img, cmap = 'gray')

                    break
                
                except OSError:
                    random_pos = get_random_position(df_matched,M1,M2, ssfr1,ssfr2)
                    invalid_file_ignored = invalid_file_ignored + 1
                    print(invalid_file_ignored," invalid files ignored")
                    print("Pos of ignored file: ",random_pos)
                    if invalid_file_ignored > 5:
                        img = 240*np.ones([500,500])
                        img[25,25] = 0
                        img[26,26] = 255

                        inset.imshow(img, cmap = 'gray')
                        break
                    continue

            

plt.show()
