import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#-----------------------------------------------------------------------------

def acquire_CIC():
    '''Takes the heart disease CSV
    turns csv into a readable pandas dataframe'''
    # Get the csv
    df = pd.read_csv('client_leads_with_outcomes.csv')
    # return pandas df
    return df

#-----------------------------------------------------------------------------

# Clean the Data
   
# set index
def set_index(df):
    '''takes in df and sets the index'''
    # set index as toekn
    df = df.set_index('token')
    # return the df
    return df

# handle outliers
def handle_outliers(df):
    '''takes in df and uses the IQR rule to remove outliers from lead_age'''
    #find out what quanitle 1 is
    q1 = df.lead_age.quantile(.25)
    #find out what quanitle 3 is
    q3 = df.lead_age.quantile(.75)
    #find out what the IQR is
    iqr = q3 - q1
    # IQR has a parameter that is a 'multiplier'
    multiplier = 1.5
    # set upper_bound
    upper_bound = q3 + (multiplier * iqr)
    # set lower_bound
    lower_bound = q1 - (multiplier * iqr)
    # remove all lead_age over the upper_bound
    df = df[df.lead_age < upper_bound]
    # this drops 6 observations 560 to 554
    return df

# dummy variable for provider
def dummy_provider(df):
    '''makes dummy features from the provider feature
    1 stands for yes and 0 stands for no'''
    # dummy provider feature
    dummy_df =  pd.get_dummies(df['provider'])
    # name the new columns (goes in order of value counts high to low)
    dummy_df.columns = ['Provider_C', 'Provider_B', 
                        'Provider_D', 'Provider_A']
    # concat the dummies to the main data frame
    df = pd.concat([df, dummy_df], axis=1)
    # return df
    return df

# completely clean
def clean_df(df):
    '''takes in df and applys funcitons set_index, handle_outliers, and dummy_provider
    converts the now cleaned dataframe into a csv'''
    # set the index
    df = set_index(df)
    # handle the outliers
    df = handle_outliers(df)
    # create dummy features from provider
    df = dummy_provider(df)
    # convert to df
    df.to_csv('clean_CIC.csv')
    
#-----------------------------------------------------------------------------

# Split the Data into Tain, Test, and Validate.

def split_CIC(df):
    '''This fuction takes in a df 
    splits into train, test, validate
    return: three pandas dataframes: train, validate, test
    '''
    # split the focused zillow data
    train_validate, test = train_test_split(df, test_size=.2, random_state=1234)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=1234)
    return train, validate, test

# Split the data into X_train, y_train, X_vlaidate, y_validate, X_train, and y_train

def split_train_validate_test(train, validate, test):
    ''' This function takes in train, validate and test
    splits them into X and y versions
    returns X_train, X_validate, X_test, y_train, y_validate, y_test'''
    X_train = train.drop(columns = ['purchase'])
    y_train = pd.DataFrame(train.purchase)
    X_validate = validate.drop(columns=['purchase'])
    y_validate = pd.DataFrame(validate.purchase)
    X_test = test.drop(columns=['purchase'])
    y_test = pd.DataFrame(test.purchase)
    return X_train, X_validate, X_test, y_train, y_validate, y_test

# Scale the Data

def scale_my_data(train, validate, test):
    scale_columns = ['lead_cost', 'lead_age', 'lead_duration']
    scaler = MinMaxScaler()
    scaler.fit(train[scale_columns])

    train_scaled = scaler.transform(train[scale_columns])
    validate_scaled = scaler.transform(validate[scale_columns])
    test_scaled = scaler.transform(test[scale_columns])
    #turn into dataframe
    train_scaled = pd.DataFrame(train_scaled)
    validate_scaled = pd.DataFrame(validate_scaled)
    test_scaled = pd.DataFrame(test_scaled)
    
    return train_scaled, validate_scaled, test_scaled

#-----------------------------------------------------------------------------
