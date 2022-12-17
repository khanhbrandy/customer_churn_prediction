import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import norm
from numpy.random import seed
from numpy import percentile
from scipy.stats import normaltest
from sklearn.utils import resample


def calculateVIF(df):
    vif = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    df_vif = pd.DataFrame(zip(df.columns, vif), columns=["feature", "vif"])
    return df_vif

def testNorm(arr):
    # Initiate the random number generator
    seed(1)
    # D'Agostino and Pearson's Test
    stat, p = normaltest(arr)
    print('Statistics={:.3f}, p={:.3f}'.format(stat, p))
    alpha = 0.05
    if p > alpha:
        print('The feature follows a normal distribution (fail to reject H0)')
    else:
        print('The feature does not follows a normal distribution (reject H0)')
        
def detectOuliers(data):
    outliers_idx = []
    # Calculate summary statistics
    _mean, _std = np.mean(data), np.std(data)
    # Identify outliers with 3*std from the Mean: 99.7%
    cut_off = _std * 3
    lower, upper = _mean - cut_off, _mean + cut_off
    for idx, ele in enumerate(data):
        if ele < lower or ele > upper:
            outliers_idx.append(idx)
    return outliers_idx
        
def upSample(X_train, y_train, target_name):
    X_train[target_name] = y_train
    df_majority = X_train[X_train[target_name]==0]
    df_minority = X_train[X_train[target_name]==1]
    df_minority_upsampled = resample(df_minority, 
                                 replace=True, # sample with replacement
                                 n_samples=len(df_majority), 
                                 random_state=42) 
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    # Split X, y again
    X = df_upsampled.drop(target_name, axis=1)
    y = df_upsampled[target_name]
    return X, y

def makeScale(X_train, X_test=None):
    columns = X_train.columns
    ss = StandardScaler()
    # Fit the scaler
    ss.fit(X_train)
    # Transform X_train
    X_train = ss.transform(X_train)
    scaled_X_train = pd.DataFrame(X_train, columns=columns)
    if X_test is not None:
        X_test = ss.transform(X_test)
        scaled_X_test = pd.DataFrame(X_test, columns=columns)
        return scaled_X_train, scaled_X_test
    else:
        return scaled_X_train
    
def makeDummies(data):
    df_object_cols = data.select_dtypes(include = 'object')
    final_df = data.select_dtypes(exclude = 'object')
    for column in df_object_cols.columns:
        dummy_features =  pd.get_dummies(df_object_cols[column], prefix=column, drop_first=True)
        final_df = pd.concat([final_df, dummy_features], axis=1)
    return final_df

def splitData(data, target_name):
    X = data.drop(target_name, axis=1)
    y = data[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


