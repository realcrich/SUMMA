# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 20:04:11 2025

@author: Collin
"""

import netCDF4 as nc
import pandas as pd
from SUMMA.SCRIPTS import *

# Define directories and filenames
ddir = 'C:/Users/Collin/SUMMA/SUMMA_shared/summaTestCases_3.0/testCases_data/inputData/fieldData/'
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
    
   # **Save DataFrame as CSV before NetCDF conversion**
    csv_path = f"{ddir}sagehen/forcing_sagehen_{elev[idx]}.csv"
    df_sagehen.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    df_sagehen = convert_timestamps_from_SNOTEL(df_sagehen,'timestamp')
                
    #copy_and_modify_netcdf(file_path, ddir+'sagehen/forcing_sagehen_'+str(elev[idx])+'.nc', ['latitude','longitude','time','LWRadAtm','SWRadAtm','airtemp','pptrate','spechum','windspd','airpres'], [lat,lon,df_sagehen['timestamp'],df_sagehen['ILWR'],df_sagehen['ISWR'],df_sagehen['TA'],df_sagehen['Prate'],df_sagehen['SpecHum'],df_sagehen['VW'],df_sagehen['Pres']])

