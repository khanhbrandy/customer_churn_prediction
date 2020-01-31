"""
Created on 2019-10-14
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
from sklearn import model_selection
from imblearn.over_sampling import SMOTE

class Preprocessor:
    def __init__(self):
        pass

    def finalize_data(self, profile_data, interest_strength):
        print('Start merging data for training...')
        start = time.time()
        data=pd.merge(interest_strength, profile_data[[
                                               'FACEBOOK_ID',
                                               'AGE',
                                            #    'AGE_RANGE',
                                               'GENDER',
                                               'PRODUCT',
                                               'LOCATION_FIVE9',
                                               'RELATIONSHIP_FIVE9',
                                               'RELATIONSHIP_CHECK',
                                               'LOCATION_MCREDIT',
                                               'LOCATION_CHECK',
                                               'RELATIONSHIP_MCREDIT',
                                               'EDUCATION',
                                               'GB'
                                               ]], how='inner', on='FACEBOOK_ID')
        data.dropna(inplace=True)
        print('Done merging data for training. Time taken = {:.1f}(s) \n'.format(time.time()-start))
        return data

    def split_data(self, data, seed, re=False):
        X, y = data.iloc[:,1:-1],data.iloc[:,-1]
        # Train-Test split
        test_size = 0.2
        X_train_o, X_test, y_train_o, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
        # Resampling
        if re:
            resam=SMOTE(random_state=seed)
            resam.fit(X_train_o, y_train_o)
            X_train, y_train = resam.fit_resample(X_train_o, y_train_o)
            X_train = pd.DataFrame(X_train, columns=X_train_o.columns)
            y_train = pd.Series(y_train)
        else:
            X_train, y_train = X_train_o,y_train_o
        return X, y, X_train, y_train, X_test, y_test
