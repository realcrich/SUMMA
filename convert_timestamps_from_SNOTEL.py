# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:40:37 2025

@author: Collin
"""

import pandas as pd
from datetime import datetime

def convert_timestamps_from_SNOTEL(df,timestamp_column_name):
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