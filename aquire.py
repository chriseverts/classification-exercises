
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import env


host = "157.230.209.171"
username = "germain_1467"
password = "8A486YgW1XMLedoP5mgqBlgw0L3NKoNC"


def get_db_url(db_name, username=username, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'



def new_titanic_data():
    '''
    This function reads in the titanic data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = 'SELECT * FROM passengers'
    df =  pd.read_sql(sql_query, get_db_url('titanic_db'))
    return df 


def get_titanic_data():
    if os.path.isfile('titanic_df.csv'):
        df = pd.read_csv('titanic_df.csv', index_col = 0)
    else:
        df = new_titanic_data()
        df.to_csv('titanic_df.csv')
    return df

##### Acquire Iris Data #####

def new_iris_data():
    sql_query = 'SELECT * FROM species JOIN measurements USING(species_id)'
    df = pd.read_sql(sql_query,get_db_url('iris_db'))
    return df

def get_iris_data():
    if os.path.isfile('iris_df.csv'):
        df = pd.read_csv('iris_df.csv', index_col=0)
    else:
        df = new_iris_data()
        df.to_csv('iris_df.csv')
    return df