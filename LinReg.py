# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:07:24 2022

@author: emmam
"""

# Import the packages and classes needed in this example:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('Dataset.csv')

data1 = pd.read_csv('train.csv')

input = x = pd.DataFrame(data1, columns= ['pH_value'])

x_train, x_test, y_train, y_test = train_test_split(data.Gender, data.pH_value, test_size = 0.2)

regr = LinearRegression()

regr.fit(np.array(x_train).reshape(-1,1), y_train)

preds = regr.predict(np.array(x_test).reshape(-1,1))

print(y_test.head())

residuals = (preds - y_test)+7

plt.hist(residuals)

#The mean squared error (MSE) tells you how close a regression line is to a set of points. 
#It does this by taking the distances from the points to the regression line (these distances 
#are the “errors”) and squaring them. The squaring is necessary to remove any negative signs. 
#It also gives more weight to larger differences. It’s called the mean squared error as you’re 
#finding the average of a set of errors. The lower the MSE, the better the forecast.
z = mean_squared_error(y_test, preds) ** 0.5

print(z)

ans = regr.predict(np.array(input).reshape(-1,1))

plt.hist(ans)

        

