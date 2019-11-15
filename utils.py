# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:16:52 2019

@author: caoa
https://github.com/fastai/course-v3/blob/master/nbs/dl1/rossman_data_clean.ipynb
"""
import pandas as pd
import numpy as np
import re

def add_datepart(df, datecolumn, drop=True, time=False):
    "Helper function that adds columns relevant to a date."
    column = df[datecolumn]
    column_dtype = column.dtype
    if isinstance(column_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        column_dtype = np.datetime64
    if not np.issubdtype(column_dtype, np.datetime64):
        df[datecolumn] = column = pd.to_datetime(column, infer_datetime_format=True)
    prefix = re.sub('[Dd]ate$', '', datecolumn)
    attributes = ['Year','Month','Week','Day','Dayofweek',
                'Dayofyear','Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start',
                'Is_year_end','Is_year_start',
                ]
    if time: 
        attributes += ['Hour']#, 'Minute', 'Second']
    for attr in attributes: 
        df[prefix + attr] = getattr(column.dt, attr.lower())
        if attr in ['Year','Dayofyear']:
            df[prefix + attr] = df[prefix + attr].astype(np.uint16)
        elif attr in ['Month','Week','Day','Dayofweek','Hour','Minute','Second']:
            df[prefix + attr] = df[prefix + attr].astype(np.uint8)

    df[prefix + 'Elapsed'] = column.astype(np.int64) // 10 ** 9
    if drop: 
        df.drop(datecolumn, axis=1, inplace=True)
    