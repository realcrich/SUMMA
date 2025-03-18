# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 20:04:11 2025

@author: Collin
"""

import netCDF4 as nc
import pandas as pd
from datetime import datetime
import numpy as np

def convert_timestamps(df,timestamp_column_name):
    """
    Convert timestamps from SNOTEL station data to days since 1990, to match SUMMA netCDF format.
    :param df: SNOTEL station dataframe.
    :param timestamp_column_name: name of timestamp column in SNOTEL dataframe.
    :return: timestamps in decimal days since 1990-01-01 00:00:00.
    """
    ref_date = datetime(1990,1,1)
    df[timestamp_column_name] = pd.to_datetime(df[timestamp_column_name])
    df[timestamp_column_name] = (df[timestamp_column_name]-ref_date).dt.total_seconds()/86400.
    return df

def calculate_specific_humidity(relative_humidity, air_temperature, pressure):
    """
    Convert relative humidity to specific humidity.
    :param relative_humidity: Relative humidity as a percentage (0-100) or fraction (0-1).
    :param air_temperature: Air temperature in Kelvin or Celsius.
    :param pressure: Air pressure in hPa.
    :param temp_in_kelvin: Boolean flag to indicate if temperature is in Kelvin (default=True).
    :return: Specific humidity (g kg-1).
    """
    # Convert inputs to numeric, forcing errors to NaN
    relative_humidity = pd.to_numeric(relative_humidity, errors='coerce')
    air_temperature = pd.to_numeric(air_temperature, errors='coerce')
    air_temperature = air_temperature - 273.15 # K --> C
    e_s = 6.112 * np.exp((17.67 * air_temperature) / (air_temperature + 243.5))  # Saturation vapor pressure in hPa
    e = relative_humidity * e_s  # Actual vapor pressure in hPa
    q = (0.622 * e) / (pressure - 0.378 * e)  # Specific humidity in g g-1
    
    return q.fillna(0)

def calculate_PRrate_from_PRsum(PRsum, time):
    """
    Calculate precipitation rate from precipitation sum.
    :param PRsum: Precipitation sum.
    :param time: Time series in days.
    :return: Precipitation rate (kg m-2 s-1).
    """
    # Convert inputs to numeric, forcing errors to NaN
    PRsum = pd.to_numeric(PRsum,errors='coerce')
    time = pd.to_numeric(time,errors='coerce'
                         )    
    dt = 3600 # s h-1
    #dt.iloc[0] = 0  # Ensuring first value is set properly
    PRsum = pd.to_numeric(PRsum, errors='coerce')  # Convert to numeric, setting invalid values to NaN
    PRrate = PRsum / dt
    #PRrate = PRrate.fillna(0)  # Replace NaN values with 0
    return PRrate

def copy_and_modify_netcdf(original_file, new_file, var_names, new_data_list):
    """
    Make a copy of SUMMA netCDF file and replace with data from SNOTEL station.
    :param original_file: SUMMA netCDF.
    :param new_file: name of new netCDF containing SNOTEL station data.
    :param new_file: list of variable names from SNOTEL station data.
    :param new_data_list: list of variables/dataframe columns to be used in making SNOTEL netCDF.
    """
    if len(var_names) != len(new_data_list):
        raise ValueError("var_names and new_data_list must have the same length.")
    
    with nc.Dataset(original_file, 'r') as src:
        with nc.Dataset(new_file, 'w', format='NETCDF4') as dst:
            # Copy dimensions
            for name, dim in src.dimensions.items():
                dst.createDimension(name, len(dim) if not dim.isunlimited() else None)
            
            # Copy variables and attributes
            for name, var in src.variables.items():
                new_var = dst.createVariable(name, var.datatype, var.dimensions, zlib=True)#, complevel=4)
                new_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                
                # Replace data if it's one of the specified variables, otherwise copy original data
                if name in var_names:
                    data = new_data_list[var_names.index(name)]
                    if name == 'PRrate':  # Ensure NaN values are set to 0 for PRrate
                        data = np.nan_to_num(data)
                    new_var[:] = data
                else:
                    new_var[:] = np.zeros(var[:].shape)
            
            # Copy global attributes
            dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

from SUMMA import *

# Define directories and filenames
ddir = 'C:/Users/Collin/Documents/SUMMA/SUMMA_shared/summaTestCases_3.0/testCases_data/inputData/fieldData/'
site_name = 'reynolds/'
file_path = f"{ddir}{site_name}forcing_above_aspen.nc"

# Load netCDF dataset
ds = nc.Dataset(file_path)

elev = [2541,2124,1962]
SNOTEL_station = [541,539,540]

for idx in range(0,3):
    # Define input file for SNOTEL data
    f_sagehen = ('C:/Users/Collin/Documents/SUMMA/SUMMA_shared/'
                 'Lisa_s_SNOWPACK_Historical_analysis/'+str(elev[idx])+'/input/SNOTEL_'+str(SNOTEL_station[idx])+'.SMET')

    # Read SNOTEL data
    df_sagehen = pd.read_csv(f_sagehen, header=None, names=['DATA'])

    lat = df_sagehen.loc[4,'DATA'].split(' = ')[-1]
    lon = df_sagehen.loc[5,'DATA'].split(' = ')[-1]

    # Extract column names from the 14th line (index 13)
    fields = df_sagehen.loc[14, 'DATA'].split(' = ')[1].split()

    # Process the data starting from line 17 (index 16)
    df_sagehen = df_sagehen.loc[16:, 'DATA'].str.split(expand=True)
    df_sagehen.columns = fields

    # Reset index
    df_sagehen.reset_index(drop=True, inplace=True)

    # Convert variables to format specified in SUMMA netCDF files
    SpecHum = calculate_specific_humidity(df_sagehen['RH'], df_sagehen['TA'], 85000)
    Pr_rate = calculate_PRrate_from_PRsum(df_sagehen['PSUM'], df_sagehen['timestamp'])

    df_sagehen['SpecHum'] = SpecHum
    df_sagehen['Prate'] = Pr_rate
    df_sagehen['Pres'] = 85000    # set constant value of 850hPa temporarily until better solution is available
    df_sagehen = convert_timestamps(df_sagehen,'timestamp')
                
    copy_and_modify_netcdf(file_path, ddir+'sagehen/forcing_sagehen_'+str(elev[idx])+'.nc', ['latitude','longitude','time','LWRadAtm','SWRadAtm','airtemp','pptrate','spechum','windspd','airpres'], [lat,lon,df_sagehen['timestamp'],df_sagehen['ILWR'],df_sagehen['ISWR'],df_sagehen['TA'],df_sagehen['Prate'],df_sagehen['SpecHum'],df_sagehen['VW'],df_sagehen['Pres']])

