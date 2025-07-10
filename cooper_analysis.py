# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:21:17 2025

@author: Collin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def slice_df_by_date(df, date_col, start_year, start_month, start_day, end_year, end_month, end_day):
    """
    Slice a DataFrame based on a start and end date.

    :param df: Pandas DataFrame containing a timestamp column.
    :param date_col: Name of the timestamp column (string).
    :param start_year: Start year (int).
    :param start_month: Start month (int).
    :param start_day: Start day (int).
    :param end_year: End year (int).
    :param end_month: End month (int).
    :param end_day: End day (int).
    :return: Sliced DataFrame between start and end dates.
    """
    # Convert the column to datetime if it's not already
    df[date_col] = pd.to_datetime(df[date_col])

    # Define start and end timestamps
    start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)

    # Slice the DataFrame based on the date range
    return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

def calculate_vpd(temp_K, RH):
    """Compute Vapor Pressure Deficit (VPD) using temperature in Kelvin and RH."""
    temp_C = temp_K - 273.15  # Convert Kelvin to Celsius
    es = 0.6108 * np.exp((17.27 * temp_C) / (temp_C + 237.3))  # Saturation vapor pressure (kPa)
    ea = es * (RH / 100)  # Actual vapor pressure (kPa)
    return es - ea  # VPD in kPa

DF_IN = pd.read_csv('C:/Users/Collin/SUMMA/SUMMA_shared/summaTestCases_3.0/testCases_data/inputData/fieldData/sagehen/forcing_sagehen_2124.csv')

DF_OUT = pd.read_csv('C:/Users/Collin/SUMMA/SUMMA_shared/sapflux/DailyTotals_922_T4N.csv')
DF_OUT2 = pd.read_csv('C:/Users/Collin/SUMMA/SUMMA_shared/sapflux/DailyTotals_922_T4S.csv')

df_in = slice_df_by_date(DF_IN, 'timestamp', 2016, 2, 1, 2019, 9, 30)
df_out = slice_df_by_date(DF_OUT, 'Date', 2016, 2, 1, 2019, 9, 30)
df_out2 = slice_df_by_date(DF_OUT2, 'Date', 2016, 2, 1, 2019, 9, 30)

nt = len(df_in)//24
df_in = df_in.iloc[:nt*24]
df_out = df_out.iloc[:nt]
df_out2 = df_out2.iloc[:nt]

# Ensure timestamp is in datetime format
df_in['timestamp'] = pd.to_datetime(df_in['timestamp'])

# Filter months
df_in = df_in[~df_in['timestamp'].dt.month.isin([10, 11, 12, 1])]

# Add VPD after filtering
df_in['VPD'] = calculate_vpd(df_in['TA'], df_in['RH'])

# Group by date
df_in['date'] = df_in['timestamp'].dt.date
df_in_day = df_in.groupby('date').mean().reset_index().rename(columns={'date': 'timestamp'})

df_out['Date'] = pd.to_datetime(df_out['Date'])
df_out = df_out[~df_out['Date'].dt.month.isin([10, 11, 12, 1])]
df_out['date'] = df_out['Date'].dt.date
df_out = df_out.groupby('date').mean().reset_index().rename(columns={'date': 'timestamp'})
df_out[:len(df_in_day)]
df_out = pd.DataFrame(df_out.mean(axis=1))[:len(df_in_day)]
df_out.insert(0,'timestamp',df_in_day['timestamp'])
df_out.rename(columns={0:'sapflux'},inplace=True)

df_out2['Date'] = pd.to_datetime(df_out2['Date'])
df_out2 = df_out2[~df_out2['Date'].dt.month.isin([10, 11, 12, 1])]
df_out2['date'] = df_out2['Date'].dt.date
df_out2 = df_out2.groupby('date').mean().reset_index().rename(columns={'date': 'timestamp'})
df_out2[:len(df_in_day)]
df_out2 = pd.DataFrame(df_out2.mean(axis=1))[:len(df_in_day)]
df_out2.insert(0,'timestamp',df_in_day['timestamp'])
df_out2.rename(columns={0:'sapflux'},inplace=True)

df_out = pd.merge(df_out, df_out2, on='timestamp', suffixes=('_N','_S'))
############## PLOTTING ################

'''
import matplotlib.colors as mcolors
import pandas as pd

# Convert to datetime and extract months
timestamps = pd.to_datetime(df_out['timestamp'])
months = timestamps.dt.month  # 1 = Jan, 12 = Dec

# Define 12 distinct colors for months
cmap = plt.cm.get_cmap('viridis', 9)  # Or use a custom list of 12 colors
norm = mcolors.BoundaryNorm(boundaries=np.arange(1,10)-0.5, ncolors=9)

for key in ['TA','TSG','ISWR','VPD']:

    # Scatter plot: colored by month
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df_in_day[key], df_out['sapflux_N'], c=months, cmap=cmap, norm=norm, marker='o')
    scS = plt.scatter(df_in_day[key], df_out['sapflux_S'], c=months, cmap=cmap, norm=norm, marker='^')

    # Colorbar setup
    cbar = plt.colorbar(sc, ticks=np.arange(1,9))
    cbar.ax.set_yticklabels(['Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep'])
    cbar.set_label('Month')

    plt.xlabel('{}'.format(key))
    plt.ylabel('Sapflux')
    plt.title('{} vs Sapflux colored by Month'.format(key))
    plt.tight_layout()
    plt.show()
'''

var_combs = [['TA','VPD'],['TA','ISWR'],['ISWR','VPD']]

import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm


# Extract month numbers for coloring
colors = pd.to_datetime(df_out['timestamp']).dt.month
# Define 12 distinct colors for months
cmap = plt.cm.get_cmap('viridis', 9)  # Or use a custom list of 12 colors
norm = BoundaryNorm(boundaries=np.arange(1,10)-0.5, ncolors=9)

for comb in var_combs:
    
    # Set up figure and 3D axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot scatter for Tower N
    scatter_n = ax.scatter(
        df_in_day[comb[0]], df_in_day[comb[1]], df_out['sapflux_N'],
        c=colors, cmap=cmap, norm=norm, s=40, alpha=0.8, label='Tower N'
    )

    # Plot scatter for Tower S
    scatter_s = ax.scatter(
        df_in_day[comb[0]], df_in_day[comb[1]], df_out['sapflux_S'],
        c=colors, cmap=cmap, norm=norm, s=40, alpha=0.8, marker='^', label='Tower S'
    )

    # Set labels and title
    ax.set_xlabel('{}'.format(comb[0]))
    ax.set_ylabel('{}'.format(comb[1]))
    ax.set_zlabel('Sapflux')
    ax.set_title('3D Scatter: {} vs {} vs Sapflux'.format(comb[0],comb[1]))

    # Colorbar for month
    cbar = plt.colorbar(scatter_n, ticks=np.arange(2, 10))
    cbar.ax.set_yticklabels(['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])
    cbar.set_label('Month')

    # Add legend
    ax.legend()

    # Function to rotate the view
    def rotate(angle):
        ax.view_init(elev=20, azim=angle)

    # Create the animation
    ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=25)

    plt.tight_layout()

    # Or to save as a video or GIF:
    ani.save("C:/Users/Collin/Desktop/{}_{}_sapflux_3d_anim.mp4".format(comb[0],comb[1]), writer="ffmpeg", fps=10)
    # ani.save("3d_rotation.gif", writer="pillow")

    plt.close(fig)
    #plt.show()