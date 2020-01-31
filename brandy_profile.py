"""
Created on 2019-10-14
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
from sklearn import preprocessing


class Profile:
    def __init__(self):
        pass

    def age_bin(self, age):    
        if pd.isna(age):
            return np.nan
        try:
            age = int(age)
        except:
            return np.nan
        if age <= 21:
            return '<= 21'
        elif age == 22:
            return '22'
        elif 23 <= age <= 27:
            return '23-27'
        elif 28 <= age <= 30:
            return '28-30'
        elif 31 <= age <= 60:
            return '31-60'
        elif age >= 61:
            return '>= 61'
        else:
            return np.nan

    def rls_convert(self, relationship):
        if pd.isna(relationship):
            return 'Unknown'
        elif relationship == 'single':
            return 'Độc thân'
        elif relationship=='married':
            return 'Kết hôn'
        elif relationship == 'divorced':
            return 'Ly hôn'
        elif relationship == 'separated':
            return 'Ly thân'
        else:
            return 'Other'
        
    def lbl_encode(self, profile_data):
        profile_data.set_index('FACEBOOK_ID', inplace=True)
        for f in profile_data.columns: 
            if f=='AGE_RANGE':
                profile_data[f]=profile_data[f].map({'<= 21':0,'22':1,'23-27':2,'28-30':3,'31-60':4,'>= 61':5})
            else:
                if profile_data[f].dtype=='O': 
                    lbl = preprocessing.LabelEncoder() 
                    lbl.fit(list(profile_data[f].values)) 
                    profile_data[f] = lbl.fit_transform(list(profile_data[f].values))
                else:
                    profile_data[f]=profile_data[f]
        profile_data=profile_data.reset_index()
        return profile_data

    def get_profile(self, profile_link):
        print('Start getting profile data for training...')
        start = time.time()
        profile_data=pd.read_csv(profile_link, header = 0, converters={'FACEBOOK_ID':str,'AGE':int,'GB':str})
        profile_data=profile_data[profile_data['GB']!='2']
        profile_data['AGE_RANGE'] = profile_data['AGE'].map(self.age_bin)
        profile_data['RELATIONSHIP_FIVE9']=profile_data['RELATIONSHIP_FIVE9'].map(self.rls_convert)
        profile_data['RELATIONSHIP_CHECK']=(profile_data['RELATIONSHIP_FIVE9']==profile_data['RELATIONSHIP_MCREDIT'])
        profile_data['LOCATION_FIVE9'].fillna('Unknown',inplace=True)
        profile_data['LOCATION_CHECK']=(profile_data['LOCATION_FIVE9']==profile_data['LOCATION_MCREDIT'])
        profile_data = self.lbl_encode(profile_data)
        print('Done getting profile data for training. Time taken = {:.1f}(s) \n'.format(time.time()-start))
        return profile_data
