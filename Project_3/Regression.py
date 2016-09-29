################################################################

import numpy as np
import pandas as pd
import patsy

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf


'''
zillow = pd.read_csv(file_path)

zillowSF = zillow[zillow["City"] == "San Francisco"].reset_index(drop=True)

print zillowSF.shape
print zillow.columns

df = zillowSF.iloc[:, 7:]

zillowSF = zillowSF.T.fillna(df.mean(axis=1)).T

print zillowSF.head()
'''

#zillow_file_path = "/Users/ga/Desktop/san_francisco/zillow_property_sales/zillow_mediansale_persqft_neighborhood.csv"
food_file_path = "/Users/ga/Desktop/san_francisco/food_inspections/food_inspections_LIVES_standard.csv"

food = pd.read_csv(food_file_path)

#print food.head()

food_new = food.iloc[:, [0, 1, 5, 6, 7, 10, 12, 16]]
print food_new.shape
#print food_new.isnull().sum()

food_na = food_new.dropna(axis = 0, how = "any").reset_index(drop=True)
print food_na.shape
#print food_na.head()

print food_na["risk_category"].unique()

mydict = {"Low Risk": float(1), "Moderate Risk": float(2), "High Risk": float(3)}

food_na["risk_category"].replace(mydict, inplace=True)

print food_na.corr()

print food_na.head()

X = food_na.iloc[:, [3, 7]]
#X = food_na.iloc[:, 7]
y = food_na.iloc[:,6]

print X.shape
print y.shape

'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "o", label = "Training Data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Test Data")

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.show()
'''

print X.head()

lm = linear_model.LinearRegression()
model = lm.fit(X,y)
predictions = lm.predict(X)
fig = plt.figure(figsize = (10,6))
plt.scatter(predictions, y, s=100, c='b', marker = '+')
plt.xlabel("Predicted Vals")
plt.ylabel("Actual Vals")
plt.show()

print "MSE:", mean_squared_error(y, predictions)
print "R2:", model.score(X,y)
print "Coeffs:", model.coef_
print "intercept:", lm.intercept_

lma = smf.ols(formula = "inspection_score ~ risk_category + business_latitude", data = food_na).fit()
print lma.params
print lma.summary()