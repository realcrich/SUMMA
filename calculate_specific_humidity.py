# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:42:45 2025

@author: Collin
"""

import pandas as pd
import numpy as np

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