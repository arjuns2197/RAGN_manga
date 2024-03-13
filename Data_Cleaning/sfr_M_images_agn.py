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

#df_matched = agn.load_matched_df()
#df_matched.to_pickle("../AGN_FIRST_files/data/df_matched_6731.pkl")


def point_exists_in_grid(df, M1, M2, ssfr1, ssfr2):

    for i in range(6731):
        if df['log_Mass'][i] > M1 and df['log_Mass'][i] < M2 and df['sSFR'][i] > ssfr1 and df['sSFR'][i] < ssfr2:
            if df['agn flux detectable'][i] == True:
                return True
        
    else:
        return False


def get_random_pifu(df,M1, M2, ssfr1, ssfr2):

    if point_exists_in_grid(df,M1, M2, ssfr1, ssfr2) == True:

        inds = []

        for i in range(6731):
            if df['log_Mass'][i] > M1 and df['log_Mass'][i] < M2 and df['sSFR'][i] > ssfr1 and df['sSFR'][i] < ssfr2:
                if df['agn flux detectable'][i] == True:
                    inds.append(i)
        
        inds = np.array(inds)

        random_ind = np.random.choice(inds)
        rand_pifu = df['plateifu'][random_ind]

        return rand_pifu


f = 30
df_matched = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6731_new.pkl")
inds_agn = df_matched['agn flux detectable'].to_numpy(dtype = bool)

ssfr = df_matched['sSFR']
M = df_matched['log_Mass']
sfr = df_matched['log_SFR_ssp']


xmin = 10
xmax = 12
ymin = -13.5
ymax = -9.5

n = int(20)


figure = plt.figure(figsize = (10,10))
ax = figure.add_gridspec(top=0.8, right=0.8).subplots()
ax_histx = ax.inset_axes([0, 1.05, 1, 0.2])
ax_histx.set_xticklabels(labels = [])
ax_histx.set_xticks([])
ax_histx.set_ylabel(r"$\rm N_{\rm AGN}$", fontsize = 0.8*f)
ax_histx.set_yticks([30,60])
ax_histx.set_yticklabels(labels = [30,60], fontsize = 0.8*f)
ax_histx.hist(M[inds_agn], bins = n, color = 'tab:gray', range = (xmin,xmax))
ax_histx.set_xlim(xmin,xmax)

ax_histy = ax.inset_axes([1.05, 0, 0.2, 1])
ax_histy.set_yticklabels(labels = [])
ax_histy.set_yticks([])
ax_histy.set_xlabel(r"$\rm N_{\rm AGN}$", fontsize = 0.8*f)
ax_histy.xaxis.set_label_position('top')
ax_histy.xaxis.tick_top()
ax_histy.hist(ssfr[inds_agn], bins = n, orientation = 'horizontal',color = 'tab:gray', range = (ymin, ymax))
ax_histy.set_xticks([20,40])
ax_histy.set_xticklabels(labels = [20,40], fontsize = 0.8*f, rotation = '-90')
ax_histy.set_ylim(ymin,ymax)


ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_xlabel(r"$\log\left(M_\star/M_\odot\right)$", fontsize = f)
ax.set_ylabel(r"$\log\left(\rm sSFR/yr^{-1}\right)$", fontsize = f)
ax.set_xticks([10.5,11,11.5])
ax.set_yticks([-13,-12,-11,-10])
ax.set_xticklabels(labels = [10.5,11,11.5],fontsize = f)
ax.set_yticklabels(labels = [-13,-12,-11,-10],fontsize = f)


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

            random_plateifu = get_random_pifu(df_matched,M1,M2, ssfr1,ssfr2)

            [plate,ifu] = random_plateifu.split('-')
            url = "https://data.sdss5.org/sas/sdsswork/sandbox/mnsa/0.3.2.analysis/pngs/" +plate +"/" + random_plateifu+"/manga-"+random_plateifu+"-Ha-O3-Grey.png"
            username = "sdss5"
            password = "panoPtic-5"

            urlresponse = requests.get(url, auth = (username,password))

            path_image = "../AGN_FIRST_files/data/manga_OIII_Ha_maps_all/" + random_plateifu + ".png"
            with open(path_image, "wb") as f:
                f.write(urlresponse.content)

            img = iio.imread(path_image)
            inset.imshow(img)

plt.savefig("../AGN_FIRST_files/results/plots/ssfr_M_dist_agn.pdf")
plt.show()






