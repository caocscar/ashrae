# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:16:17 2019

@author: caoa
"""
import pandas as pd
import numpy as np
import os
from utils import add_datepart
import time
    
pd.options.display.max_rows = 30
pd.options.display.max_columns = 10

t0 = time.time()
wdir = 'data'

meter_dict = {0: 'electricity',
              1: 'chilledwater',
              2: 'steam',
              3:' hotwater',
              }

#%%
ashrae_dtypes = {'site_id': np.uint8,
               'building_id': np.uint16,
               'square_feet': np.uint32,
               'meter': np.uint8,
               'meter_reading': np.float64,
               'air_temperature': np.float32,
               'cloud_coverage': np.float32,
               'dew_temperature': np.float32,
               'precip_depth_1_hr': np.float32, #np.int16 < 0
               'sea_level_pressure': np.float32,
               'wind_direction': np.float32, # np.uint16
               'wind_speed': np.float32, 
               }
bldg = pd.read_csv(os.path.join(wdir,'building_metadata.csv'), dtype=ashrae_dtypes)

#%%
# use mode to fill NAs
bldg['floor_count'].fillna(1, inplace=True)
bldg['floor_count'] = bldg['floor_count'].astype(np.uint8)
# use median to fill NAs
bldg_yblt = bldg.groupby(['site_id'])['year_built'].median()
bldg_yblt.fillna(round(bldg_yblt.median(),0), inplace=True)
bldg_yblt = bldg_yblt.astype(int)
bldg['year_built'] = bldg.apply(lambda x: bldg_yblt[x['site_id']], axis=1).astype(np.uint16)
print('bldg done')
t1 = time.time()

#%% Process training part
train = pd.read_csv(os.path.join(wdir,'train.csv'), dtype=ashrae_dtypes)
train = train[train['meter_reading'] > 0]
rows = train.shape[0]
weather_train = pd.read_csv(os.path.join(wdir,'weather_train.csv'), dtype=ashrae_dtypes)
weather_train.loc[weather_train['precip_depth_1_hr'] < 0, 'precip_depth_1_hr'] = 0
# use forward fill and backfill to fill NAs
weather_train.sort_values(['site_id','timestamp'], inplace=True)
weather_train = weather_train.fillna(method='ffill').fillna(method='bfill')
weather_train['wind_direction'] = weather_train['wind_direction'].astype(np.uint16)
weather_train['precip_depth_1_hr'] = weather_train['precip_depth_1_hr'].astype(np.uint16)
weather_train['cloud_coverage'] = weather_train['cloud_coverage'].astype(np.uint8)
weather_train = bldg.merge(weather_train, how='inner', on='site_id')
# ignore missing weather values
train = train.merge(weather_train, how='inner', on=['building_id','timestamp'])
train.sort_values(['building_id','timestamp'], inplace=True)
train.fillna(method='ffill', inplace=True)
add_datepart(train, 'timestamp', drop=False, time=True)
train.drop(['timestamp','timestampYear'], axis=1, inplace=True)
print('train wrangling done')
t2 = time.time()
assert train.shape[0] == 18257718
train.to_pickle('train_clean.pkl') #459 seconds
t3 = time.time()
train.to_pickle('train_clean.pkl.gz')
t4 = time.time()
train.to_csv('train_clean.csv', index=False)
t5 = time.time()
print('train saving done')
assert 1 > 2

#%%
train = None
weather_train = None
#%% Process testing part
test = pd.read_csv(os.path.join(wdir,'test.csv'), dtype=ashrae_dtypes)
rows = test.shape[0]
weather_test = pd.read_csv(os.path.join(wdir,'weather_test.csv'), dtype=ashrae_dtypes)
weather_test.sort_values(['site_id','timestamp'], inplace=True)
weather_test = weather_test.fillna(method='ffill').fillna(method='bfill')
weather_test['wind_direction'] = weather_test['wind_direction'].astype(np.uint16)
weather_test['precip_depth_1_hr'] = weather_test['precip_depth_1_hr'].astype(np.uint16)
weather_test['cloud_coverage'] = weather_test['cloud_coverage'].astype(np.uint8)
weather_test = bldg.merge(weather_test, how='inner', on='site_id')
test = test.merge(weather_test, how='left', on=['building_id','timestamp'])
test.sort_values(['building_id','timestamp'], inplace=True)
test.fillna(method='ffill', inplace=True)
add_datepart(test, 'timestamp', drop=False, time=True)
print('test wrangling done')
t5 = time.time()
assert test.shape[0] == rows
test.to_pickle('test_clean.pkl.gz')
t6 = time.time()
print('test saving done')
