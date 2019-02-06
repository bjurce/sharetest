# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:19:08 2019

@author: barba
"""
import os
os.chdir(r"C:\Users\barba\Desktop\appdoc\myRFMpackage")

#%%

import pandas as pd
import numpy as np
from datetime import date
import os

"""
####################################################################
Part 1: Data preparation
####################################################################
"""
#%%Import the data and transform the TransDate column

transactions = pd.read_csv('transactions.csv')
transactions["TransDate"] = pd.to_datetime(transactions["TransDate"],
                                            format="%d.%m.%Y",
                                            utc=True,
                                            dayfirst=True)
#check the data
transactions.info()
transactions.head()
transactions.describe()
#%%
"""
####################################################################
Part 2: Aggregation of variables
####################################################################
"""
#%%
#Save the latest transaction as an object in your variable explorer ####
maxDate = transactions["TransDate"].max()

#Create a new dataframe called rfm. ####
#that includes the customer ID, as well as the measures for purchase recency, frequency and monetary value.

#recency = difference between latest transaction and "today"
transactions["Dayslasttransactions"] = (maxDate - transactions["TransDate"]).dt.days#get the days only with the datetime.days
#Frequency = number of transactions
transactions["frequency"] = transactions.groupby("Customer")["Customer"].transform("count")

#Monetary is simply the average of transaction per customer, so we can include it in our aggregate function:
rfm = transactions.groupby('Customer', as_index=False).agg({
        "frequency":'max',
        "Dayslasttransactions":"min",
        "PurchAmount":"mean"})

#Rename the columns properly:
rfm = rfm.rename(columns={
        "Dayslasttransactions":"recency",
        "PurchAmount":"monetary"})


#Check the structure of the new table and ensure that all the variables are numeric. ####
rfm.info()
#%%
"""
####################################################################
Part 3: Calculation of RFM Scores
####################################################################
"""
#%%

# Define scores from 1 to 3 for each of the previously specified measures.
# (Hint: Consider using the pd.cut() command.)

#Use the cut2() function  ####
#in order to transform recency, frequency, and monetary value into scores from 1 to 3.

rfm_scores = rfm.copy()

#we need to add plus one otherwise bins=0-2
#here we need to invert the scale
rfm_scores['recency'] = pd.qcut(rfm_scores['recency']*-1,q=3,labels=False, duplicates='drop') + 1

#here we need to use a rank function first because qcut cannot put the same value in different bins.
rfm_scores['frequency'] = pd.qcut(rfm_scores['frequency'].rank(method='first'),q=3,labels=False, duplicates='raise') + 1

rfm_scores['monetary'] = pd.qcut(rfm_scores['monetary'],q=3,labels=False, duplicates='drop') + 1
#check if the distribution is as expected.
rfm_scores.describe()
#%%
"""
####################################################################
Part 4: Calculation of Overall RFM Score
####################################################################
"""
#%%
#Calculate the unweighted overall score for all customers, i.w. R 33.33%, F 33.33%, M 33.33%.
#Calculate a weighted overall score which weighs frequency more heavily, i.e. R 20%, F 60%, M 20%.
#Calculate a weighted overall score which weighs recency more heavily, i.e. R 60%, F 20%, M 20%.

#Calculation of overall RFM scores ####
rfm_scores['overall'] = rfm_scores[['recency','frequency','monetary']].mean(axis=1) #unweighted RFM score
rfm_scores['weighted1'] = rfm_scores['recency']*0.2 + rfm_scores['frequency']*0.6 + rfm_scores['monetary']*0.2 #weighted RFM score (frequency)
rfm_scores['weighted2'] = rfm_scores['recency']*0.6 + rfm_scores['frequency']*0.2 + rfm_scores['monetary']*0.2 #weighted RFM score (recency)

#Divide all customers in 3 distinct RFM groups by rounding the overall RFM score. ####
#Do this for all the three RFM scores, you previously calculated.

#Copy the required columns of the scores DataFrame
rfm_groups = rfm_scores[['Customer','overall','weighted1','weighted2']]

#Use an apply function to round an all columns at once
rfm_groups[['overall', 'weighted1', 'weighted2']] = rfm_groups[['overall', 'weighted1', 'weighted2']].applymap(round)
#Alternatively, we can also do it for each column:
rfm_groups['overall'] = rfm_groups['overall'].round()
rfm_groups['weighted1'] = rfm_groups['weighted1'].round()
rfm_groups['weighted2'] = rfm_groups['weighted2'].round()

#Get best customers
rfm_scores.loc[rfm_scores['overall'].max() == rfm_scores['overall']]

#%%

"""
####################################################################
Exercise 5: Create a RFM function
####################################################################
"""

def calculateRFMscores(data, weight_recency=1, weight_frequency=1, weight_monetary=1):
     
    # Ensure that the weights add up to one
    weight_recency2 = weight_recency/sum([weight_recency, weight_frequency, weight_monetary])
    weight_frequency2 = weight_frequency/sum([weight_recency, weight_frequency, weight_monetary])
    weight_monetary2 = weight_monetary/sum([weight_recency, weight_frequency, weight_monetary])
    
    # RFM measures
    max_Date=max(data["TransDate"])
    rfm=data.groupby("Customer", as_index=False).agg({"TransDate":"max",#recency = difference between latest transaction and "today"
                    "Quantity": "count", #frequency = number of transactions
                    "PurchAmount":"mean"}) #monetary = average amount spent per transaction
    #rename the colums
    rfm.rename(columns = {"TransDate":"Recency", "Quantity":"Frequency", "PurchAmount": "Monetary"}, inplace=True)
    #recency is defined as max.date - last purchase
    rfm["Recency"]=max_Date-rfm["Recency"]
    #make sure recency is numeric
    rfm["Recency"]=rfm["Recency"].dt.days

    # RFM scores
    rfm_scores = rfm.copy()

    #we need to add plus one otherwise bins=0-2
    #here we need to invert the scale
    rfm_scores['Recency'] = pd.qcut(rfm_scores['Recency']*-1,q=3,labels=False, duplicates='drop') + 1

    #here we need to use a rank function first because qcut cannot put the same value in different bins.
    rfm_scores['Frequency'] = pd.qcut(rfm_scores['Frequency'].rank(method='first'),q=3,labels=False, duplicates='raise') + 1

    rfm_scores['Monetary'] = pd.qcut(rfm_scores['Monetary'],q=3,labels=False, duplicates='drop') + 1

    # Overall RFM score
    rfm_scores["Finalscore"]=rfm["Frequency"]*weight_frequency2+rfm["Monetary"]*weight_monetary2+rfm["Recency"]*weight_recency2
    
    # RFM group
    rfm_scores["group"]=round(rfm_scores["Finalscore"])
    
    return rfm_scores
#%%
calculateRFMscores(transactions,20,20,60)
