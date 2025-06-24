from pdb import line_prefix
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from AGN_FIRST_code import agn_functions as agn

from scipy.special import gammaincc
import astropy
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.table import vstack
import pandas as pd
from scipy import integrate
from scipy import interpolate
import emcee
import time
#log_lambda_max = 2

f = 30

df = pd.read_pickle("../AGN_FIRST_files/data/df_matched_6255_new_ERs.pkl")

Q_mask = (df['sSFR'] > -16) & (df['sSFR'] < -11.5)
SF_mask = df['sSFR'] > -11.5

# Define mass bins
mass_bins = [10, 10.4, 10.8, 11.2, 11.6, 12]

# Prepare figure
plt.figure(figsize=(10, 6))

# Scatter plot
plt.scatter(df.loc[Q_mask, 'log_Mass'], df.loc[Q_mask, 'sSFR'], color='red', s=50, label='Quiescent', alpha = 0.25)
plt.scatter(df.loc[SF_mask, 'log_Mass'], df.loc[SF_mask, 'sSFR'], color='blue', s=50, label='Star forming', alpha = 0.25)

# AGN overlay
agn_mask = df['agn flux detectable'] == True
plt.scatter(df.loc[agn_mask, 'log_Mass'], df.loc[agn_mask, 'sSFR'], color='black', s=105, label='Detected AGN', zorder=3)

# Draw boxes for bins
for i in range(len(mass_bins) - 1):
    x0 = mass_bins[i]
    x1 = mass_bins[i + 1]

    # Box for Q region
    plt.gca().add_patch(plt.Rectangle(
        (x0, -16), x1 - x0, 4.5,  # height from -16 to -11.5
        fill=False, edgecolor='red', linestyle='--', linewidth = 1.5
    ))

    # Box for SF region
    plt.gca().add_patch(plt.Rectangle(
        (x0, -11.5), x1 - x0, 2,  # height from -11.5 to 0 (assuming max sSFR is near 0)
        fill=False, edgecolor='blue', linestyle='--', linewidth = 1.5
    ))

# Draw shaded boxes for bins
for i in range(len(mass_bins) - 1):
    x0 = mass_bins[i]
    x1 = mass_bins[i + 1]

    # Shaded box for Q region
    plt.gca().add_patch(plt.Rectangle(
        (x0, -16), x1 - x0, 4.5,  # height from -16 to -11.5
        fill=True, facecolor='red', alpha=0.01, edgecolor='red', linestyle='--'
    ))

    # Shaded box for SF region
    plt.gca().add_patch(plt.Rectangle(
        (x0, -11.5), x1 - x0, 2,  # height from -11.5 to 0
        fill=True, facecolor='blue', alpha=0.01, edgecolor='blue', linestyle='--'
    ))

# Labels and legend
plt.xlabel(r'$\log_{\rm 10} \left( M_{\star}/ M_{\odot}\right)$', fontsize = f)
plt.ylabel(r'$\log_{\rm 10} \left( sSFR/ yr^{-1}\right)$', fontsize = f)
plt.legend(loc = 'lower right', fontsize = f)
plt.tight_layout()
plt.xlim(9.8, 12.2)
plt.xticks([10,10.5, 11,11.5,12], fontsize = 0.75*f)
plt.ylim(-19,-9)
plt.yticks([-19,-17, -15, -13,-11, -9], fontsize = 0.75*f)
plt.show()