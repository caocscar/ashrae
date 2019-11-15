# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:53:39 2019

@author: caoa
"""

import pickle
import pandas as pd
import numpy as np

forrest = pickle.load(open('RF4.pkl','rb'))
df = pd.read_pickle('test_clean.pkl') #23 seconds

#%%
DV = 'meter_reading'
rm_vars = ['meter_reading','timestamp','primary_use']
rm_vars.append('row_id')
rm_vars.remove(DV)
onehot = pd.get_dummies(df['primary_use'])
df.drop(rm_vars, axis=1, inplace=True)
test = df.merge(onehot, left_index=True, right_index=True)
abc = test.sample(10)
yhat = forrest.predict(abc)

#%%
y = []
for tree in forrest.estimators_:
    y.append(tree.predict(abc))
fgh = np.vstack(y).T
predictions = np.mean(fgh, axis=1)

