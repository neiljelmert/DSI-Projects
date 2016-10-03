################################################################
################################################################

import datetime
from datetime import datetime
import numpy as np
import pandas as pd
import patsy
import re
import glob
from cycler import cycler
from ipywidgets import *
from IPython.display import display

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

# ZILLOW
###################################################################################
'''

zillow_file_path = "/Users/ga/Desktop/san_francisco/zillow_property_sales/zillow_mediansale_persqft_neighborhood.csv"
zillow = pd.read_csv(file_path)

zillowSF = zillow[zillow["City"] == "San Francisco"].reset_index(drop=True)

print zillowSF.shape
print zillow.columns

df = zillowSF.iloc[:, 7:]

zillowSF = zillowSF.T.fillna(df.mean(axis=1)).T

print zillowSF.head()
'''


###################################################################################
###################################################################################
###################################################################################
###################################################################################

# FOOD
###################################################################################
'''

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
'''


###################################################################################
###################################################################################
###################################################################################
###################################################################################

# FIRE
###################################################################################

'''

fire_path = "/Users/ga/Desktop/san_francisco/fire_data/fire_incidents.csv"

fire = pd.read_csv(fire_path)

print fire.shape
#print fire.columns

fire.drop(["Incident Number", "Exposure Number", "Address", "Incident Date",
           "Call Number", "Zipcode", "Station Area", "Box", "First Unit On Scene",
           "Number of Alarms", "Supervisor District", "Neighborhood  District",
           "Location", "City", "Action Taken Secondary", "Action Taken Other",
           "Property Use", "Area of Fire Origin", "Ignition Cause",
           "Ignition Factor Primary", "Ignition Factor Secondary",
           "Heat Source", "Item First Ignited", "Human Factors Associated with Ignition",
           "Detector Type", "Detector Operation", "Detector Failure Reason",
           "Automatic Extinguishing Sytem Type", "Automatic Extinguishing Sytem Perfomance",
           "Automatic Extinguishing Sytem Failure Reason", "Number of Sprinkler Heads Operating",
           "Detector Effectiveness", "Automatic Extinguishing System Present",
           "Fire Spread", "No Flame Spead", "Mutual Aid", "Battalion"], axis = 1, inplace=True)


fire["Detector Alerted Occupants"] = fire["Detector Alerted Occupants"].fillna(-1)
fire["Estimated Property Loss"] = fire["Estimated Property Loss"].fillna(np.mean(fire["Estimated Property Loss"]))
fire["Estimated Contents Loss"] = fire["Estimated Contents Loss"].fillna(np.mean(fire["Estimated Contents Loss"]))
fire["Detectors Present"] = fire["Detectors Present"].fillna(0)
fire["Structure Type"] = fire["Structure Type"].fillna(-1)
fire["Structure Status"] = fire["Structure Status"].fillna(-1)
fire["Floor of Fire Origin"] = fire["Floor of Fire Origin"].fillna(np.mean(fire["Floor of Fire Origin"]))
fire["Number of floors with minimum damage"] = fire["Number of floors with minimum damage"].fillna(0)
fire["Number of floors with significant damage"] = fire["Number of floors with significant damage"].fillna(0)
fire["Number of floors with heavy damage"] = fire["Number of floors with heavy damage"].fillna(0)
fire["Number of floors with extreme damage"] = fire["Number of floors with extreme damage"].fillna(0)


# clean detectors
detectors_dict = {"1 -present": 1, "-": 0, "n -not present": 0,
                  "1 present": 1, "n none present": 0, "u -undetermined": -1,
                  "u undetermined": -1
                  }
fire["Detectors Present"] = fire["Detectors Present"].replace(detectors_dict)


# clean detector alerted occupants
det_alert_dict = {"2 detector did not alert occupants": 0,
                  "1 - detector alerted occupants": 1,
                  "2 - detector did not alert occupants": 0,
                  "u - unknown": -1,
                  "1 detector alerted occupants": 1,
                  "u unknown": -1,
                  "-": -1
                  }
fire["Detector Alerted Occupants"] = fire["Detector Alerted Occupants"].replace(det_alert_dict)

# clean structure type
struct_type_dict = {"1 -enclosed building": 1, "3 -open structure": 3,
                    "-": -1, "2 -fixed portable or mobile structure": 2,
                    "1 enclosed building": 1, "0 -structure type, other": 0,
                    "5 -tent": 5, "6 -open platform": 6, "7 -underground structure work areas": 7,
                    "8 -connective structure": 8, "0 structure type, other": 0, "4 -air supported structure": 4,
                    "3 open structure": 3, "6 open platform": 6, "8 connective structure": 8,
                    "2 fixed portable or mobile structure": 2, "7 underground structure work area": 7,
                    "4 air-supported structure": 4
                    }
fire["Structure Type"] = fire["Structure Type"].replace(struct_type_dict)


# clean structure status
struct_stat_dict = {"2 -in normal use": 2, "-": -1, "3 -idle, not routinely used": 3,
                    "5 -vacant and secured": 5, "2 in normal use": 2,
                    "6 -vacant and unsecured": 6, "0 -other": 0, "1 -under construction": 1,
                    "4 -under major renovation": 4, "1 under construction": 1, "4 under major renovation": 4,
                    "u -undetermined": -1, "7 -being demolished": 7, "u undetermined": -1,
                    "5 vacant and secured": 5, "0 building status, other": 0, "6 vacant and unsecured": 6,
                    "3 idle, not routinely used": 3, "7 being demolished": 7
                    }
fire["Structure Status"] = fire["Structure Status"].replace(struct_stat_dict)


#remove nulls
fire = fire.dropna(axis = 0, how = "any").reset_index(drop=True)
#print fire.isnull().sum()


def mysplit(col):
    mydict = {}
    for elmnt in col.unique():
        s = elmnt.replace(" - ", " ").split(" ")
        for code in s:
            if code.isdigit() and code == s[0]:
                mydict[elmnt] = int(code)
    col.replace(mydict, inplace = True)
    return col

mysplit(fire["Primary Situation"])
mysplit(fire["Action Taken Primary"])


#fire = fire.iloc[0:int(round(.08*fire.shape[0])),:]
#print fire.shape


fire["Alarm DtTm"] = fire["Alarm DtTm"].apply(lambda date: datetime.strptime(date, "%m/%d/%Y %I:%M:%S %p"))
fire["Arrival DtTm"] = fire["Arrival DtTm"].apply(lambda date: datetime.strptime(date, "%m/%d/%Y %I:%M:%S %p"))
fire["Close DtTm"] = fire["Close DtTm"].apply(lambda date: datetime.strptime(date, "%m/%d/%Y %I:%M:%S %p"))


# calculate time differences
fire["Alarm Arrival Diff"] = fire["Arrival DtTm"] - fire["Alarm DtTm"]
fire["Alarm Arrival Diff"] = fire["Alarm Arrival Diff"].apply(lambda time: int(time.total_seconds()))
fire["Alarm Close Diff"] = fire["Close DtTm"] - fire["Alarm DtTm"]
fire["Alarm Close Diff"] = fire["Alarm Close Diff"].apply(lambda time: int(time.total_seconds()))
fire["Arrival Close Diff"] = fire["Close DtTm"] - fire["Arrival DtTm"]
fire["Arrival Close Diff"] = fire["Arrival Close Diff"].apply(lambda time: int(time.total_seconds()))

#print fire["Structure Status"].unique()

fire = fire[fire["Primary Situation"] != "y -"]
fire = fire[fire["Primary Situation"] != "n/a -"]
fire = fire[fire["Primary Situation"] != "cr -"]
fire = fire[fire["Primary Situation"] != "25* -"]
fire = fire[fire["Action Taken Primary"] != "-"]


#print fire["Primary Situation"].unique()
#print fire["Action Taken Primary"].unique()


print fire.dtypes
print fire.shape
print fire.corr()

sns.set(context="paper", font="monospace")
#fire = sns.load_dataset("fire", header=[0, 1, 2], index_col=0)
corrmat = fire.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, square = True)
plt.xticks(rotation='vertical')
plt.yticks(rotation = 0)
sns.plt.show()

X = fire[["Number of floors with significant damage", "Number of floors with heavy damage",
                  "Number of floors with extreme damage", "Alarm Close Diff"]]
y = fire[["Fire Injuries"]]

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

lma = smf.ols(formula = "Fire Fatalities ~ Alarm Close Diff", data = X).fit()
print lma.params
print lma.summary()

'''
###################################################################################
###################################################################################
###################################################################################
###################################################################################
# concatenate all assessor csvs from 2007 to 2015
# assessor data: contains information about yearly house assessments
# includes attributes of each house: bed no., room no., value, etc.

path = "/Users/ga/Desktop/san_francisco/assessor_office/"

files = []
for i in range(2007, 2015):
    file_name = "assessor_data_" + str(i) + "_" + str(i+1) + ".csv"
    files.append(file_name)

list_ = []
#frame = pd.DataFrame()
for f in files:
    allFiles = glob.glob(path + f)

    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None)
        list_.append(df)
        #print file_, df.columns

assessor = pd.concat(list_)

##################################################################################
# CLEAN, CLEAN, CLEAN...

assessor.drop(["BASELOTSFX", "BUILTIN", "CONSTTYPE", "EXEMPTYPE",
               "KITCHEN", "LOTCODE", "REMARKFLA", "REPRISDATE",
               "RP1CLACDE", "RP1DOCNO1", "RP1NBRCDE",
               "RP1STACDE", "ZONE", "PROPLOC", "RP1PRCLID",
               "OWNRPRCNT", "RECURRSALD", "RP1TRACDE"], axis=1, inplace=True)

assessor["BASELOT"] = assessor["BASELOT"].fillna(np.mean(assessor["BASELOT"]))
assessor["FBA"] = assessor["FBA"].fillna(np.mean(assessor["FBA"]))
assessor["LAREA"] = assessor["LAREA"].fillna(np.mean(assessor["LAREA"]))

bed_mean = assessor["BEDS"][assessor["BEDS"] != 0].mean()
assessor["BEDS"] = assessor["BEDS"].replace(0, bed_mean)
assessor = assessor.dropna(axis=0, how="any").reset_index(drop=True)

yr_dict = {95: 1995, 13: 1913, 9185: 1985, 6889: 1989, 6573: 1973,
           8687: 1987, 24: 1924, 21: 1921, 22: 1922, 41: 1941}

assessor["YRBLT"].replace(yr_dict, inplace=True)
assessor = assessor[assessor["YRBLT"] < 2017]
assessor = assessor[assessor["YRBLT"] > 1600]

valcols = ["RP1FXTVAL", "RP1IMPVAL", "RP1LNDVAL", "RP1PPTVAL"]
assessor["VALUE"] = assessor[valcols].sum(axis=1)
assessor = assessor.iloc[:, ~assessor.columns.isin(valcols)]

# print assessor.shape (1483188, 20)

for col in ["UNITS", "STOREYNO", "SQFT", "ROOMS", "BEDS", "BATHS"]:
    assessor = assessor[assessor[col] != 0]

assessor = assessor[assessor["VALUE"] > 20000].reset_index(drop=True)

# print assessor.shape (1154771, 20)

###################################################################################
# visualize correlation among features

corrmat = assessor.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, square = True)
plt.xticks(rotation='vertical')
plt.yticks(rotation = 0)
sns.plt.show()

####################################################################################
# create the predictors and target
# we will predict home value based on the features strongest correlated to value:
# baths, rooms, sqft, storeyno, units

# Hypothesis: The value of a home depends on the above features

X = assessor[["RP1DOCNO2", "BATHS", "ROOMS", "STOREYNO", "SQFT", "UNITS"]]
y = assessor[["VALUE"]]

####################################################################################
#scale the predictors and target

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
y_norm = scaler.fit_transform(y)


####################################################################################
# Linear Regression

lm = linear_model.LinearRegression()
model = lm.fit(X_norm,y_norm)
predictions = lm.predict(X_norm)
fig = plt.figure(figsize = (10,6))
plt.scatter(predictions, y_norm, s=100, c='b', marker = '+')
plt.xlabel("Predicted Vals")
plt.ylabel("Actual Vals")
sns.plt.show()

print "MSE:", mean_squared_error(y_norm, predictions)
print "R2:", model.score(X_norm,y_norm)
print "Coeffs:", model.coef_
print "intercept:", lm.intercept_


###################################################################################
# SMF OLS Summary

X2 = assessor[["VALUE", "UNITS", "STOREYNO", "SQFT", "ROOMS", "FRONT",
              "FBA", "BEDS", "BATHS"]]

lma = smf.ols(formula = 'VALUE ~ UNITS + STOREYNO + SQFT + ROOMS + BATHS',
              data = X2).fit()
print lma.params
print lma.summary()


#####################################################################################
# Train Test Split using Linear Regression

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, train_size = 0.75)

lm_tts = linear_model.LinearRegression()
model = lm_tts.fit(X_train, y_train)
predictions_tts = lm_tts.predict(X_test)

plt.scatter(y_test, predictions_tts)
plt.xlabel("Truth")
plt.ylabel("Pred")
sns.plt.show()

print " TTS score:", model.score(X_test, y_test)

###################################################################################
# K-Fold Cross-Validation using Linear Regression

lm_cv = linear_model.LinearRegression()
scores = cross_val_score(lm_cv, X_norm, y_norm, cv = 5)
print "Cross-validated score:", scores

predictions_cross = cross_val_predict(lm_cv, X_norm, y_norm, cv = 5)
plt.scatter(y_norm, predictions_cross)
plt.xlabel("Truth")
plt.ylabel("Pred")
sns.plt.show()

accuracy = metrics.r2_score(y_norm, predictions_cross)
print "Cross-Predicted Accuracy:", accuracy
print "K-Fold Mean Scores:", np.mean(scores)

###################################################################################
# Visualizing the Ridge, Lasso, and Elastic on our dataset

simple_feature_names = ["UNITS", "STOREYNO", "SQFT", "ROOMS", "BATHS"]

def ridge_coefs(X, y, alphas):
    coefs = []
    ridge_reg = Ridge()
    for a in alphas:
        ridge_reg.set_params(alpha=a)
        ridge_reg.fit(X, y)
        coefs.append(ridge_reg.coef_[0])
    return coefs

r_alphas = np.logspace(0, 10, 200)
r_coefs = ridge_coefs(X_norm, y_norm, r_alphas)

def coef_plotter(alphas, coefs, feature_names, to_alpha, regtype='ridge'):
    amin = np.min(alphas)
    amax = np.max(alphas)
    alphas = [a for a in alphas if a <= to_alpha]
    coefs = coefs[0:len(alphas)]
    colors = sns.color_palette("husl", len(coefs[0]))
    fig = plt.figure()
    fig.set_size_inches(18, 5)
    ax1 = fig.add_subplot(121)
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.axvline(to_alpha, lw=2, ls='dashed', c='k', alpha=0.4)
    ax1.plot(alphas, coefs, lw=2)
    ax1.set_xlabel('alpha', fontsize=20)
    ax1.set_ylabel('coefficients', fontsize=20)
    if regtype == 'ridge':
        ax1.set_xscale('log')
    ax1.set_xlim([amin, amax])
    ax1.set_title(regtype + ' coef paths\n', fontsize=20)
    ymin, ymax = ax1.get_ylim()
    ax2 = fig.add_subplot(122)
    ax2.bar(range(1, len(feature_names) + 1), coefs[-1], align='center', color=colors)
    ax2.set_xticks(range(1, len(feature_names) + 1))
    ax2.set_xticklabels(feature_names, rotation=65, fontsize=12)
    ax2.set_ylim([ymin, ymax])
    ax2.set_title(regtype + ' predictor coefs\n', fontsize=20)
    ax2.set_xlabel('coefficients', fontsize=20)
    ax2.set_ylabel('alpha', fontsize=20)
    plt.show()

### RIDGE

def ridge_plot_runner(log_of_alpha=0):
    coef_plotter(r_alphas, r_coefs, simple_feature_names, 10**log_of_alpha, regtype='ridge')

interact(ridge_plot_runner, log_of_alpha=(0.0,10.0,0.05))

### LASSO

def lasso_coefs(X, Y, alphas):
    coefs = []
    lasso_reg = Lasso()
    for a in alphas:
        lasso_reg.set_params(alpha=a)
        lasso_reg.fit(X, Y)
        coefs.append(lasso_reg.coef_)
    return coefs

l_alphas = np.arange(0.01, 1, 0.0025)
l_coefs = lasso_coefs(X_norm, y_norm, l_alphas)

def lasso_plot_runner(alpha=0):
    coef_plotter(l_alphas, l_coefs, simple_feature_names, alpha, regtype='lasso')

interact(lasso_plot_runner, alpha=(0.01,1,0.0025))

### ELASTIC

def elasticnet_coefs(X, Y, alphas):
    coefs = []
    enet_reg = ElasticNet()
    for a in alphas:
        enet_reg.set_params(alpha=a, l1_ratio=0.05)
        enet_reg.fit(X, Y)
        coefs.append(enet_reg.coef_)
    return coefs

enet_alphas = np.arange(0.001, 2.0, 0.005)
enet_coefs = elasticnet_coefs(X_norm, y_norm, enet_alphas)

def enet_plot_runner(alpha=0):
    coef_plotter(enet_alphas, enet_coefs, simple_feature_names, alpha, regtype='elastic net')

interact(enet_plot_runner, alpha=(0.001,2.0,0.005))

