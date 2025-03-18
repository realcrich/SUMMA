# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:45:36 2025

@author: Collin
"""

import netCDF4 as nc
import numpy as np

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