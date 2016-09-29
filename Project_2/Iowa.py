#########################################################################

import numpy as np
import scipy.stats as stats
import pandas as pd
import datetime
from datetime import datetime

iowa_file = '/Users/ga/Desktop/DSI-SF-3_repo/DSI-SF-3/datasets/iowa_liquor/Iowa_Liquor_sales_sample_10pct.csv'
iowa = pd.read_csv(iowa_file)
#print "Dataframe is of size: " + str(iowa.values.nbytes / 10**6) + "MB"

#print iowa.shape (270955, 18)
#print iowa.columns
#print iowa.describe()

#begin cleaning
iowa = iowa.dropna(axis=0, how = "any").reset_index(drop=True)
iowa["State Bottle Cost"] = iowa["State Bottle Cost"].map(lambda x: float(x.lstrip("$")))
iowa["State Bottle Retail"] = iowa["State Bottle Retail"].map(lambda x: float(x.lstrip("$")))
iowa["Sale (Dollars)"] = iowa["Sale (Dollars)"].map(lambda x: float(x.lstrip("$")))
iowa["County Number"] = iowa["County Number"].map(lambda x: int(x))
iowa["Category"] = iowa["Category"].map(lambda x: int(x))
iowa["Bottles Sold"] = iowa["Bottles Sold"].map(lambda x: float(x))
iowa["Date"] = iowa["Date"].apply(lambda date: datetime.strptime(date, "%m/%d/%Y"))

iowa = iowa.sort_values(by = "Date").reset_index(drop=True)

store_num_list = iowa["Store Number"].unique()
for store_number in store_num_list:

    dates_of_sale = iowa[iowa["Store Number"] == store_number]["Date"].reset_index()["Date"]
    died_index = dates_of_sale.shape[0] - 1
    born = dates_of_sale[0]
    died = dates_of_sale[died_index]

    if born > datetime(2015, 01, 10):
        iowa = iowa[iowa["Store Number"] != store_number]

    if died < datetime(2015, 12, 20):
        iowa = iowa[iowa["Store Number"] != store_number]

print iowa.shape #(231256, 18)

# I removed all rows in which a store number first appears after January 10, 2015, \n
# or disappears before December 20, 2015 -- giving a two-sided 10 day forgiveness \n
# due to circumstances such as holiday vacation



iowa_15 = iowa[iowa["Date"] <= datetime(2015, 12, 31)] #(188065, 18)
iowa_15 = iowa_15[iowa_15["Date"] >= datetime(2015, 01, 01)]

#print iowa_2015.shape

store_num_list_15 = iowa_15["Store Number"].unique()

profit = {}
store_2015_sales = {}
for store_number in store_num_list_15:
    bot_sold = iowa_15[iowa_15["Store Number"] == store_number][["Bottles Sold"]].values
    sale = iowa_15[iowa_15["Store Number"] == store_number][["Sale (Dollars)"]].values
    retail = iowa_15[iowa_15["Store Number"] == store_number]["State Bottle Retail"].values
    zipper_sales = zip(bot_sold, sale)
    zipper_retail = zip(bot_sold, retail)
    #print zipper
    tot_sales = int(sum(x*y for x, y in zipper_sales))
    retail_tot = int(sum(x*y for x, y in zipper_retail))

    #print tot_sales

    store_2015_sales[store_number] = tot_sales
    profit[store_number] = tot_sales - retail_tot

print store_2015_sales
print profit