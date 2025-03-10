# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 13:09:26 2025

@author: Collin
"""

import netCDF4 as nc
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

def convert_seconds_to_datetime(nc_time_var):
    """
    Convert a NetCDF time variable (seconds since 1990-01-01) to datetime timestamps.

    :param nc_time_var: NetCDF time variable (numpy array or list).
    :return: NumPy array of datetime objects.
    """
    ref_date = datetime(1990, 1, 1)

    # Ensure input is a NumPy array
    nc_time_var = np.array(nc_time_var, dtype=float)

    # Mask invalid values (NaN, inf, negative)
    valid_mask = np.isfinite(nc_time_var) & (nc_time_var >= 0)

    # Initialize an array with None values
    datetime_array = np.full(nc_time_var.shape, None, dtype=object)

    # Convert valid values to datetime
    datetime_array[valid_mask] = [ref_date + timedelta(seconds=t) for t in nc_time_var[valid_mask]]

    return datetime_array


def write_netcdf_to_csv(netcdf_paths, output_dir):
    """
    Reads multiple NetCDF files and converts their contents into CSV files.
    
    Parameters:
        netcdf_paths (list): List of paths to input NetCDF files.
        output_dir (str): Directory to save the output CSV files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for netcdf_path in netcdf_paths:
        # Open the NetCDF file
        dataset = nc.Dataset(netcdf_path, mode='r')
        
        headers = {}
        length_groups = {}
        
        # Categorize variables based on length
        for var in dataset.variables:
            data = dataset.variables[var][:].flatten()
            data_length = len(data)
            
            if var == 'time':
                t = convert_seconds_to_datetime(dataset.variables[var][:])
            
            elif data_length == 1:
                headers[var] = data[0]  # Store single-value variables as headers
            else:
                if data_length not in length_groups:
                    length_groups[data_length] = {}
                length_groups[data_length][var] = data
        
        # Use the first key in length_groups for CSV creation
        if length_groups:
            first_length = next(iter(length_groups))
            df = pd.DataFrame(length_groups[first_length])
            
            df.insert(5,'time',t)
            df['time'] = df['time'].dt.round('H')
            
            # Convert headers into a DataFrame and prepend them to the main DataFrame
            if headers:
                header_df = pd.DataFrame([headers] * len(df))
                df = pd.concat([header_df, df], axis=1)
            
            # Define CSV filename based on NetCDF filename
            csv_filename = os.path.splitext(os.path.basename(netcdf_path))[0] + ".csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            print(f"CSV file saved at: {csv_path}")
        
        # Close the dataset
        dataset.close()

elev = [2541,2124,1962]

for lev in elev:
    
    # name input files using different elevations
    f_BB = 'C:/Users\Collin/Documents/SUMMA/SUMMA_shared/summaTestCases_3.0/output/Default par, Init cond_/{}/figure07/vegImpactsTranspire_ballBerry_timestep.nc'.format(lev)
    f_Jar = 'C:/Users\Collin/Documents/SUMMA/SUMMA_shared/summaTestCases_3.0/output/Default par, Init cond_/{}/figure07/vegImpactsTranspire_jarvis_timestep.nc'.format(lev)
    f_SR = 'C:/Users\Collin/Documents/SUMMA/SUMMA_shared/summaTestCases_3.0/output/Default par, Init cond_/{}/figure07/vegImpactsTranspire_simpleResistance_timestep.nc'.format(lev)
    
    # create list of file names to be used in function
    fnames = [f_BB,f_Jar,f_SR]
    
    # specifiy output directory location
    out_dir = 'C:/Users/Collin/Documents/SUMMA/SUMMA_shared/summaTestCases_3.0/output/Default par, Init cond_/{}/figure07/as_csv'.format(lev)
    
    write_netcdf_to_csv(fnames,out_dir)