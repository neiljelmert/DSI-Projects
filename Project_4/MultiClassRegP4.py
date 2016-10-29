# import shiz

import pandas as pd
import numpy as np
import patsy

from sklearn.linear_model import Ridge, Lasso, LinearRegression, RidgeCV, LassoCV,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import statsmodels.formula.api as sm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import itertools

###############################################################
# read data

business = pd.read_csv("/Users/ga/Desktop/Yelp/yelp_arizona_data/businesses_small_parsed.csv")
reviews = pd.read_csv("/Users/ga/Desktop/Yelp/yelp_arizona_data/reviews_small_nlp_parsed.csv")
users = pd.read_csv("/Users/ga/Desktop/Yelp/yelp_arizona_data/users_small_parsed.csv")
tips = pd.read_csv("/Users/ga/Desktop/Yelp/yelp_arizona_data/tips_small_nlp_parsed.csv")
checkins = pd.read_csv("/Users/ga/Desktop/Yelp/yelp_arizona_data/checkins_small_parsed.csv")

###############################################################
# take every 20th row (big)

bus_samp = business.iloc[::20,:]
rev_samp = reviews.iloc[::20,:]
users_samp = users.iloc[::20,:]
tips_samp = tips.iloc[::20,:]
check_samp = checkins.iloc[::20,:]

###############################################################
# make some copies for later

bus_samp_copy = bus_samp.copy()
rev_samp_copy = rev_samp.copy()
users_samp_copy = users_samp.copy()
tips_samp_copy = tips_samp.copy()
check_samp_copy = check_samp.copy()

#tips_samp_copy = tips_samp_copy.select(lambda x: x in bus_samp_copy["business_id"])
#rev_samp_copy = rev_samp_copy.select(lambda x: x in bus_samp_copy["business_id"])

tips_samp_copy = tips_samp_copy.loc[tips_samp_copy["business_id"].isin(bus_samp_copy["business_id"].tolist())]
rev_samp_copy = rev_samp_copy.loc[rev_samp_copy["business_id"].isin(bus_samp_copy["business_id"].tolist())]

###############################################################
# make a cleaning function for businesses
bus_cat_dict = dict(zip(
	bus_samp_copy["business_id"], bus_samp_copy["categories"]))

print 'MwmXm48K2g2oTRe7XmssFw' in bus_cat_dict.keys()

def bus_clean(bus, col_list):
	# customization
	bus["variable"] = bus["variable"].apply(lambda x: x.split(".")[1:])
	bus = bus[bus.astype(str)["variable"] != "[]"]
	#bus["city"] = bus["city"].apply(lambda x: 1 if x == "Las Vegas" else 0)

	# pair up variable and value, make into a new column
	pairs = []
	for pair in zip(bus["variable"], bus["value"]):
	    if len(pair[0]) > 1:
	        pair = ' '.join(pair[0]) + ": " + pair[1]
	        pairs.append(pair)
	    else:
	        pair = str(pair[0][0]) + ": " + str(pair[1])
	        pairs.append(pair)
	bus["var_val"] = pairs

	bus = bus.drop(col_list, axis=1) # drop unwanted cols
	dum = pd.get_dummies(bus.iloc[:,1:])
	bus = pd.concat([bus["business_id"], dum], axis=1)
	return bus

###############################################################

def tr_clean(tips, rev):

	tips = tips.drop(["likes", "date", "user_id"], axis=1)
	rev = rev.drop(["votes.cool", "review_id", "votes.funny",
		"stars", "date", "votes.useful", "user_id"], axis=1)
	tips.columns = [col + "_tip" for col in tips.columns]
	tips = tips.rename(
		columns = {"user_id_tip": "user_id", 
					"business_id_tip": "business_id"})

	tips_reviews = pd.merge(tips, rev, how = "outer", on = "business_id")
	return tips_reviews

###############################################################

col_list = ["name", "latitude", "longitude", "neighborhoods", 
			"variable", "value", "city", "review_count"]

X = pd.merge(
	bus_clean(bus_samp_copy, col_list), 
	tr_clean(tips_samp_copy, rev_samp_copy), 
	how = "outer", on = "business_id")


X["categories"] = [bus_cat_dict[bid] for bid in X["business_id"]]
X = X.drop("business_id", axis=1)
X = X.fillna(0)

categories = X["categories"].apply(eval)

###############################################################

X.drop("categories", axis=1)
cat_list = list(itertools.chain(categories.tolist()))
y = MultiLabelBinarizer().fit_transform(cat_list)

print X.shape, y.shape

OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

































