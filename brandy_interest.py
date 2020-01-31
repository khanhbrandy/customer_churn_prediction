"""
Created on 2019-12-16
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
from copy import copy as make_copy
from sklearn import preprocessing

class Interest:
    def __init__(self):
        pass

    def get_interest(self, url):
        data=pd.read_csv(url, header = 0, converters={0:str,1:str})
        data = data.head(100) ### For testing purpose !!!!!
        data = self.interest_standardize(data, std=False)
        fbids=np.array(data['FACEBOOK_ID'])
        return data, fbids
    
    def interest_standardize(self, interest_data, ids=None, std=False):
        interest_data=interest_data.iloc[:,1:]
        if ids==None:
            interest_data=interest_data
        else:
            interest_data=interest_data[ids]
        if std:
            interest_data.set_index('FACEBOOK_ID', inplace=True)
            scaler = preprocessing.StandardScaler()
            interest_data_stded=scaler.fit_transform(interest_data.values)
            df_interest_data_stded=pd.DataFrame(interest_data_stded, index=interest_data.index,columns=interest_data.columns)
            df_interest_data_stded.reset_index(inplace=True)
        else:
            df_interest_data_stded=interest_data
        return df_interest_data_stded

    def data_merge(self, level_list, merge=True):
        print('Start getting interest data for training...')
        start = time.time()
        for level_dic in level_list:
            level_dic['data'], level_dic['fbid'] = self.get_interest(level_dic['link'])
        print('Done getting interest data for training. Time taken = {:.1f}(s) \n'.format(time.time()-start))
        data_final = make_copy(level_list[0]['data'])
        dfs = [level_list[i]['data'] for i in range(1,5)]
        if merge:
            for df in dfs:
                data_final = data_final.merge(df, on=['FACEBOOK_ID'], how='left')
                interest_strength = data_final.fillna(0)
        else:
            interest_strength=level_list[4]['data'].fillna(0)
        def level_convert(fbid):
            if fbid in set(level_list[4]['fbid']):
                    return 5
            else:
                if fbid in set(level_list[3]['fbid']):
                    return 4
                else:
                    if fbid in set(level_list[2]['fbid']):
                        return 3
                    else:
                        if fbid in set(level_list[1]['fbid']):
                            return 2
                        else:
                            return 1
        interest_strength['INTEREST_LEVEL'] = interest_strength['FACEBOOK_ID'].map(level_convert)
        # Get ids 
        sum_ids = interest_strength.sum(axis=0)
        sum_ids[sum_ids != 0]
        ids = sum_ids[sum_ids != 0].index
        return interest_strength, ids

    
