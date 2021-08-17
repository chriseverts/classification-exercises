from pydataset import data
import seaborn as sns
import pandas as pd
import numpy as np
import os
from env import host, username, password, get_db_url
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def prep_iris(df):
    '''
    takes in a dataframe of the iris dataset as it is acquired and returns a cleaned dataframe
    arguments: df: a pandas DataFrame with the expected feature names and columns
    return: clean_df: a dataframe with the cleaning operations performed on it
    '''
    df = df.drop(columns=['species_id', 'measurement_id'])
    df = df.rename(columns={'species_name': 'species'})
    dummy_df = pd.get_dummies(df[['species']], dummy_na=False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

def impute_mode(train, validate, test):
    '''
    impute mode for embark_town
    '''
    imputer = SimpleImputer(strategy='most_frequent', missing_values=None)
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test

def prep_titanic_data(df):
    '''
    takes in a dataframe of the titanic dataset as it is acquired and returns a cleaned dataframe
    arguments: df: a pandas DataFrame with the expected feature names and columns
    return: train, test, split: three dataframes with the cleaning operations performed on them
    '''
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embarked', 'class', 'age', 'passenger_id'])
    train, test = train_test_split(df, test_size=0.2, random_state=1349, stratify=df.survived)
    train, validate = train_test_split(train, train_size=0.7, random_state=1349, stratify=train.survived)
    train, validate, test = impute_mode(train, validate, test)
    dummy_train = pd.get_dummies(train[['sex', 'embark_town']], drop_first=[True,True])
    dummy_validate = pd.get_dummies(validate[['sex', 'embark_town']], drop_first=[True,True])
    dummy_test = pd.get_dummies(test[['sex', 'embark_town']], drop_first=[True,True])
    train = pd.concat([train, dummy_train], axis=1)
    validate = pd.concat([validate, dummy_validate], axis=1)
    test = pd.concat([test, dummy_test], axis=1)
    train = train.drop(columns=['sex', 'embark_town'])
    validate = validate.drop(columns=['sex', 'embark_town'])
    test = test.drop(columns=['sex', 'embark_town'])
    return train, validate, test

def prep_titanic_data_for_logit(df):
    '''
    takes in a dataframe of the titanic dataset as it is acquired and returns a cleaned dataframe
    arguments: df: a pandas DataFrame with the expected feature names and columns
    return: train, test, split: three dataframes with the cleaning operations performed on them
    '''
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embarked', 'class', 'passenger_id'])
    df.age = df.age.fillna(value=df.age.mean())
    train, test = train_test_split(df, test_size=0.2, random_state=1349, stratify=df.survived)
    train, validate = train_test_split(train, train_size=0.7, random_state=1349, stratify=train.survived)
    train, validate, test = impute_mode(train, validate, test)
    dummy_train = pd.get_dummies(train[['sex', 'embark_town']], drop_first=[True,True])
    dummy_validate = pd.get_dummies(validate[['sex', 'embark_town']], drop_first=[True,True])
    dummy_test = pd.get_dummies(test[['sex', 'embark_town']], drop_first=[True,True])
    train = pd.concat([train, dummy_train], axis=1)
    validate = pd.concat([validate, dummy_validate], axis=1)
    test = pd.concat([test, dummy_test], axis=1)
    train = train.drop(columns=['sex', 'embark_town'])
    validate = validate.drop(columns=['sex', 'embark_town'])
    test = test.drop(columns=['sex', 'embark_town'])
    return train, validate, test