#########################################################################

import numpy as np
import scipy.stats as stats
import pandas as pd
import datetime
from datetime import datetime

iowa_file = '/Users/ga/Desktop/DSI-SF-3_repo/DSI-SF-3/datasets/iowa_liquor/Iowa_Liquor_sales_sample_10pct.csv'
iowa = pd.read_csv(iowa_file)


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

#total yearly income
total = iowa_15.groupby("Store Number")[["Sale (Dollars)"]].sum()


#yearly profit
def mul_cols(group):
    total = group[["Sale (Dollars)"]].sum()
    loss = np.sum(group["State Bottle Cost"] * group["Bottles Sold"])
    return total - loss
profit = iowa_15.groupby("Store Number").apply(mul_cols)

#print profit


#top counties profit by volume
profit["Store Number"] = profit.index
my_merge = iowa_15.merge(profit, on = "Store Number", how='left')
my_merge["Yearly Profit"] = my_merge["Sale (Dollars)_y"]
my_merge["Sale (Dollars)"] = my_merge["Sale (Dollars)_x"]

def div_cols(group):
    volume = group[["Volume Sold (Gallons)"]].sum()
    prof_by_vol = np.divide(group["Yearly Profit"], volume)
    return prof_by_vol.sum()
my_merge.groupby("County").apply(div_cols).sort_values(ascending=False)

#Polk, Johnson, Story, ...

