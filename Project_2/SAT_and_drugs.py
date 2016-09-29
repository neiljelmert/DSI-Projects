# Project 2: SAT Scores

import scipy.stats as stats
import csv
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot, array
plt.interactive(False)

####################################
############ TASK 1 ############
sat_filepath = '/Users/ga/Desktop/DSI-SF-3_repo/DSI-SF-3/datasets/state_sat_scores/sat_scores.csv'

# Subtask 1.1

# dictionary which allows access to value given state and rate, math, \n
# or verbal (thought this was cooler =] )
mydict1 = {}
mydict2 = {}
with open(sat_filepath) as sat_csv:
    sat_read = csv.reader(sat_csv)
    headers = next(sat_read)[0:]
    for row in sat_read:
        mydict1[row[0]] = {header: value for header, value in zip(headers, row[0:])}

    print mydict1["WV"]["Math"] #gives 512


# dictionary wanted from exercise
sat_read_dict = csv.DictReader(open(sat_filepath))
for row in sat_read_dict:
    for column, value in row.iteritems():
        mydict2.setdefault(column, []).append(value)

print mydict2


# Subtask 1.2

sat = pd.read_csv(sat_filepath)
sat_from_dict = pd.DataFrame.from_dict(mydict2)
print sat.dtypes
print sat_from_dict.dtypes

# the difference is that sat_from_dict df considers all values to be objects, \n
# whereas the sat df considers Rate, Verbal, Math to be int64, and State to be \n
# object

# Subtask 1.3

print sat.head(10)
# The State, Math, and Verbal columns are obvious. The Rate column I believe \n
# refers to the percent of students who scored over the verbal/math combo

####################################
############ TASK 2 ############
# Create data dictionary

sat_dict = {column: (sat[column].shape[0], sat[column].dtype) for column in sat.columns}
print sat_dict

####################################
############ TASK 3 ############
# Plot the data using seaborn
# Subtask 3.1


sat_long = pd.melt(sat, id_vars = "State")

g = sns.FacetGrid(sat_long,
                  col = "variable",
                  gridspec_kws = {"width_ratios": [5, 5, 5]},
                  sharex = False,
                  sharey = False)

g.map(sns.distplot, "value", bins = 15, kde = False)
sns.plt.show()


# Subtask 3.2

sns.pairplot(sat)
sns.plt.show()

# Pairplot gives us a visualization of how well correlated each column \n
# in our dataset is

####################################
############ TASK 4 ############
# Plot the data using built-in pandas functions
# Subtask 4.1
# Plot stacked histogram Verbal Math

sat[["Verbal", "Math"]].plot.hist(stacked = True, bins = 15)
sns.plt.show()

# Subtask 4.2
# Plot Verbal Math on same chart using boxplots

sat[["Verbal", "Math"]].plot.box()
sns.plt.show()

# Subtask 4.3
# Plot Verbal Math Rate appropriately on boxplot chart

sat_norm = sat[["Verbal", "Math"]].apply(lambda x: x/np.max(x))
sat_norm["Rate"] = sat["Rate"]/100
#print sat_norm.head()

names = ["Verbal/max", "Math/max", "Rate/100"]
sat_norm[["Verbal", "Math", "Rate"]].plot.box()
plt.xticks([1, 2, 3], names)
sns.plt.show()

# Intuitively I've "normalized" the Math and Verbal columns by dividing out \n
# the max for each column, and additionally normalized the Rate in terms of \n
# percentage; thus all column values now lie between 0 and 1.


####################################
############ TASK 5 ############
# Create and examine subsets of data
# Subtask 5.1

verb_mean = np.mean(sat["Verbal"])
#print "mean", verb_mean
states_above_mean = sat[sat["Verbal"] > verb_mean]["State"]
print "Number of states above mean", states_above_mean.count()
# this tells me the distribution is fairly even

# Subtask 5.2

verb_med = np.median(sat["Verbal"])
#print "median", verb_med
states_above_median = sat[sat["Verbal"] > verb_med]["State"]
print "Number of states above median", states_above_median.count()

# this tells me there is hardly any skew since mean ~ median

# Subtask 5.3

sat["Diff Verb Math"] = sat["Verbal"] - sat["Math"]

# Subtask 5.4

VM_max_diff_plus = sat.sort_values("Diff Verb Math", ascending = False)["State"][:10].reset_index(drop=True)
print VM_max_diff_plus.head(3)

VM_max_diff_neg = sat.sort_values("Diff Verb Math")["State"][:10].reset_index(drop=True)
print VM_max_diff_neg.head(3)

####################################
############ TASK 6 ############
# Examine Summary Stats
# Subtask 6.1

print sat.corr()
# The correlation is a number between -1 and +1 that measures \n
# how close the relationship between two variables is to being linear \n
# with +1 signifying a perfectly positive correlation (direct relationship) \n
# and -1 signifying a perfectly negative correlation (inverse relationship)

# Subtask 6.2

print sat.describe()
# The "count" row gives us the number of non-unique values in each column
# The "mean" row gives us the mean of each column
# The "std" row gives us the standard deviation of each column
# The "min" row gives us the minimum value within each column
# The "25%" row gives us the first quartile of each column
# The "50%" row gives us the second quartile (or median) of each column
# The "75%" row gives us the third quartile of each column
# The "max" row gives us the maximum value within each column

# Subtask 6.3

print sat.cov()
# The correlation matrix is a "normalized" covariance matrix. This is \n
# useful since covariances are hard to compare; by "quotienting" out the \n
# diversity and scale of both covariates, we get a value between -1 and 1 \n
# which aids better comparison.
# Corr(x,y) = Cov(x,y)/sqrt(Var(x)*Var(y))
# The correlation matrix tells us how well each column vector correlates with the other; \n
# the covariance matrix tells us the mean value of the product of the deviations \n
# of two variates from their respective means.


#############################################################################
# Project 2: DRUG BY AGE Data

####################################
############ TASK 7 ############

# Subtask 7.1

drug_data_filepath = '/Users/ga/Desktop/DSI-SF-3_repo/DSI-SF-3/datasets/drug_use_by_age/drug-use-by-age.csv'
drugs_data = pd.read_csv(drug_data_filepath)

# Subtask 7.2

print drugs_data.columns
print drugs_data.shape
#print drugs_data.describe()
# The data set contains 17 rows of age ranges with corresponding number of samples \n
# per age (the "n" column), with multiple drug use frequencies (where a value of 5, for instance, means \n
# the subject on average uses the drug once every 5 days)

# Subtask 7.3

# What are the highest correlated features?

sns.set(context = "paper", font = "monospace")
corrmat = drugs_data.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax = 0.8, square = True)
sns.plt.show()

unstacked = corrmat.unstack()
so = unstacked.sort_values()

print so

print drugs_data[["marijuana-use", "pain-releiver-use"]].corr()
sns.pairplot(drugs_data[["marijuana-use", "pain-releiver-use"]])
sns.plt.show()

print drugs_data[["marijuana-frequency", "pain-releiver-frequency"]].corr()
sns.pairplot(drugs_data[["marijuana-frequency", "pain-releiver-frequency"]])
sns.plt.show()

# Surprisingly, pain-reliever-use and marijuana-use are among the most highly correlated
# However, their frequencies of use are unrelated, suggesting \n
# that the distribution of use might be entirely chance

# Other questions I have are: what is n? Is this the number of people in each age group \n
# who have used the drug? Is the frequency how often they do it per week or month or year? \n
# Having answers to these questions would help tighten my specific question.

####################################
############ TASK 8 ############
# Subtask 8.1

print "Cov for SAT", np.cov(sat[["Math", "Verbal", "Rate"]])

# Refer to Subtask 6 for answers to this question; the questions are the same

# Subtask 8.2

# Covariance matrix is calculated as C(i,j) = Cov(X_i, X_j)

# Correlation matrix is the covariance matrix of the standardized random \n
# variables X_i / STD(X_i)

# Correlation has values between -1 and +1; easier to make comparisons
# Covariance is "limitless"; more difficult to compare

# Subtask 8.3

def cov_mat(m):
    X = array(m, dtype=float)
    X -= X.mean(axis=0)
    N = X.shape[0]
    fact = float(N - 1)
    print dot(X, X.T.conj()) / fact


x = [-2.1, -1]
y = [3, 1.1]
M = np.vstack((x,y))
cov_mat(M)

# Same for corr, just multiply by appropriate constants

####################################
############ TASK 9 ############

Q1 = sat["Rate"].quantile(0.25)
Q3 = sat["Rate"].quantile(0.75)

#print Q1 - 1.0*(Q3 - Q1)
#print Q3 + 1.0*(Q3 - Q1)
#print "MEAN", np.mean(sat["Rate"])
#print "STD", np.std(sat["Rate"])

for rate in sat["Rate"]:
    if rate < Q1 - 1.5*(Q3 - Q1) or rate > Q3 + 1.5*(Q3 - Q1):
        #print abs(rate - np.mean(sat["Rate"]))/np.std(sat["Rate"])
        print rate

# NO OUTLIERS

#sns.distplot(sat["Rate"], bins = 50)
#sns.plt.show()

####################################
############ TASK 10 ############
# Subtask 10.1

print "SPEAR", stats.spearmanr(sat["Verbal"], sat["Math"])
print "PEARS", stats.pearsonr(sat["Verbal"], sat["Math"])

# Pearson correlation measures the linear relationship between two continuous vars
# Spearman correlation measures the monotonic relationship between two continuous vars
# Spearman correlation is gotten as follows:
       # rank the columns
       # compute the covariance of the rank variables
       # compute the Pearson correlation coefficient
       # find the stds of the rank variables
       # compute cov(rank_x, rank_y)/std(rank_x)*std(rank_y)

# Subtask 10.2

RatePerc = np.array([stats.percentileofscore(sat["Rate"], score = x) for x in sat["Rate"]])
sat["RatePerc"] = RatePerc

print "CA PERCENTILE", sat[sat["State"] == "CA"]["RatePerc"]

# Percentile ranks the data based on how much of the data falls belows the percentage

# Subtask 10.3

drugs_alc_PS = np.array([stats.percentileofscore(drugs_data["alcohol-frequency"], score = x) for x in drugs_data["alcohol-frequency"]])


sns.distplot(drugs_data["alcohol-frequency"], bins = 20)
sns.plt.show()
sns.distplot(drugs_alc_PS, bins = 15)
sns.plt.show()