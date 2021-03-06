# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 09:30:52 2020

@author: Mateusz
"""
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Path  of the file to read
air_pollution_file_path = 'C:\\Users\\Mateusz\\Desktop\\Cambridge\\Dissertation\\Data\\datasets\\vic.csv'

# Fill in the line below to read the file into a variable home_data
data = pd.read_csv(air_pollution_file_path)
#%%

# Print summary statistics in next line
print(data.describe())

print(data.columns)

y = data.ScaledValue


print(y.describe)

data_features = ['ground_level', 'raw_platform_depth', 'platform_depth', 'longitude(x)', 'latitude(y)', 'Line_ID', 'Entry_week', 'age', 'mean_air_polution ', 'Temp_C', 'Rain_mm', 'Humidity', 'Pressure',
       'Wind_m/s']

X = data[data_features]

print(X.describe())

print(X.head())
#%%
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

#max_leaf_nodes=600, random_state = 1
data_model = LinearRegression()
#%%
data_model.fit(train_X, train_y)
#%%
print("Making predictions for the following 5 situations:")
print(X.head())
print("The predictions are")
print(data_model.predict(X.head()))

#%%

predictions = data_model.predict(val_X)
print (predictions)


#%%

rmse = np.sqrt(mean_squared_error(val_y, predictions))
R = r2_score(val_y, predictions)
MAPE = np.mean(np.abs((val_y - predictions) / val_y)) * 100

print("RMSE: %f" % (rmse))

print("R^: ", R )

print("MAPE: ", MAPE )
#%%

#plt.scatter(val_X, val_y,  color='black')
plt.plot(val_X['age'], predictions, color='blue', linewidth=3)


plt.xticks(())
plt.yticks(())

plt.show()




















