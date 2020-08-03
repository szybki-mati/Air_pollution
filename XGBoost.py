
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:00:23 2020

@author: Mateusz
"""
from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

# Path  of the file to read
air_pollution_file_path = 'C:\\Users\\Mateusz\\Desktop\\Cambridge\\Dissertation\\Data\\datasets\\LUexportMay2016v1_PM.csv'  #LUexportMay2016v1_PM.csv

# Fill in the line below to read the file into a variable home_data
data = pd.read_csv(air_pollution_file_path)
#%%

# Print summary statistics in next line
print(data.describe())

print(data.columns)

y = data.ScaledValue

#%%
print(y.describe())

data_features = ['ground_level', 'raw_platform_depth', 'platform_depth', 'longitude(x)', 'latitude(y)', 'Line_ID', 'Entry_week', 'age', 'mean_air_polution ', 'Temp_C', 'Rain_mm', 'Humidity', 'Pressure',
       'Wind_m/s']

X = data[data_features]

print(X.describe())

print(X.head())
#%%
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


my_model = XGBRegressor()
#%%
my_model.fit(train_X,  train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)
#%%
print("Making predictions for the following 5 situations:")
print(X.head())
print("The predictions are")
print(my_model.predict(X.head()))

#%%

predictions = my_model.predict(val_X)
#print (predictions)

#%%

#print((mean_squared_error(val_y, )

rmse = np.sqrt(mean_squared_error(val_y, predictions))
R = r2_score(val_y, predictions)
MAPE = np.mean(np.abs((val_y - predictions) / val_y)) * 100

print("RMSE: %f" % (rmse))

print("R^: ", R )

print("MAPE: ", MAPE )


#%%
xgb.plot_importance(my_model)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
#%%
xgb.plot_tree(my_model,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()
#xgb.plot_tree(my_model,num_trees=0)
#plt.rcParams['figure.figsize'] = [50, 10]
#plt.show()















