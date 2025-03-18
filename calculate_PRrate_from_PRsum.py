# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:43:59 2025

@author: Collin
"""

import pandas as pd

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