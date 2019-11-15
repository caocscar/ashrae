# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:15:55 2019

@author: caoa
"""
from comet_ml import Experiment
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import time
from sklearn.externals import joblib

experiment = Experiment(api_key="ttcF3s2M7Y8vmLq6flwrfjVsK",
                        project_name="ashrae", 
                        workspace="caocscar",
                        )
pd.options.display.max_rows = 46

t1 = time.time()
df = pd.read_pickle('train_clean.pkl') #23 seconds
print(f'Read train pickle: {(time.time()-t1):.1f} seconds')
df['floor_count'] = df['floor_count'].astype(int)

#%%
onehot = pd.get_dummies(df['primary_use'])
train = df.merge(onehot, left_index=True, right_index=True)
rm_vars = ['meter_reading','timestamp','primary_use']
x_train = train.drop(rm_vars, axis=1)
y_train = train['meter_reading']

#%%
t3 = time.time()
trees = 5
depth = 4
forrest = RandomForestRegressor(n_estimators = trees,
                                criterion = 'mse',
                                max_features = 'auto',
                                random_state = 42619,
                                n_jobs = 4,
                                oob_score = True,
                                max_depth = depth,
                                )
forrest.fit(x_train, y_train)
t4 = time.time()
print(f'Fit RF Regressor: {(t4-t3)/60:.1f} minutes')
experiment.log_metric("oob_score", forrest.oob_score_)
joblib.dump(forrest, f'RF{trees}_D{depth}.jbl')

#%%
t5 = time.time()
df = pd.read_pickle('test_clean.pkl') #23 seconds
print(f'Read test pickle: {(time.time()-t5):.1f} seconds')

DV = 'meter_reading'
rm_vars = ['meter_reading','timestamp','primary_use']
rm_vars.append('row_id')
rm_vars.remove(DV)
onehot = pd.get_dummies(df['primary_use'])
df.drop(rm_vars, axis=1, inplace=True)
test = df.merge(onehot, left_index=True, right_index=True)
assert df.shape[0] == test.shape[0]
t7 = time.time()
yhat = forrest.predict(test)
t8 = time.time()
print(f'Predictions: {(t8-t7)/60:.1f} minutes')

#%%
t9 = time.time()
sub = pd.DataFrame(yhat)
sub.reset_index(inplace=True)
sub.columns = ['row_id','meter_reading']
sub['meter_reading'] = sub['meter_reading'].round(4)
sub.to_csv(f'submission_RF{trees}_D{depth}.csv', index=False)
t10 = time.time()
print(f'Submission: {(t10-t9)/60:.1f} minutes')
