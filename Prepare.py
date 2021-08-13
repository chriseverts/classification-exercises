#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
import aquire
iris_df = aquire.get_iris_data()

def clean_data(df):
    iris_df = df.drop(columns = ['species_id', 'measurement_id'])
    iris_df = df.rename(columns=({'species_name': 'species'}))
    iris_df = pd.get_dummies(df, prefix = ['species'])
    return iris_df

def prep_iris(df):
    iris_df = clean_data(df)
    return iris_df


# In[ ]:




