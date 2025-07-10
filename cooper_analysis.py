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

df_in = slice_df_by_date(DF_IN, 'timestamp', 2016, 2, 1, 2019, 9, 30)
df_out = slice_df_by_date(DF_OUT, 'Date', 2016, 2, 1, 2019, 9, 30)

nt = len(df_in)//24
df_in = df_in.iloc[:nt*24]
df_out = df_out.iloc[:nt]

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
df_out = pd.DataFrame(df_out.mean(axis=1))
df_out.insert(0,'timestamp',df_in_day['timestamp'])
df_out.rename(columns={0:'sapflux'},inplace=True)

plt.figure()
plt.scatter(df_in_day['TA'],df_out)
plt.show()