# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:52:41 2020

@author: 
"""

import pandas as pd
import os
import psutil
process = psutil.Process(os.getpid())
import time
from probables import (CountMinSketch)
import sys
import json

in_data=[]

for root, subdirs, files in os.walk('C:/Users/Teoi/Desktop/json files'):
    for file in files:
        with open (file,errors='ignore',encoding="UTF-8") as infile:
            for line in infile:
                in_data.append(json.loads(line))

################## for hashtags ##################

start_time1 = time.time()

df=pd.DataFrame(in_data)
df2=df[['entities']]
df3=pd.concat([df2.drop(['entities'], axis=1), df2['entities'].apply(pd.Series)], axis=1)

hashtag_df=df3.explode('hashtags')
hashtag_df.drop(hashtag_df.columns.difference(['hashtags']), 1, inplace=True)
hashtag_df=pd.concat([hashtag_df.drop(['hashtags'], axis=1), hashtag_df['hashtags'].apply(pd.Series)], axis=1)
hashtag_df.drop(hashtag_df.columns.difference(['text']), 1, inplace=True)
hashtag_df['Count']=1
hashtag_df=hashtag_df.dropna()
hashtag_df_final=hashtag_df.groupby(['text']).sum().sort_values(by=['Count'],ascending=False)

end_time1 = time.time()

cms=CountMinSketch(width=128,depth=8)
hashtag_df=hashtag_df['text'].reset_index()

hash_list=hashtag_df['text'].tolist()
hash_list_len=len(hash_list)

hash_series=pd.Series(hash_list)

start_time2 = time.time()

for i in range(0,hash_list_len):
    cms.add(str(hash_series[i]))

end_time2 = time.time()

cms.check('WorldCup2014')
cms.check('WorldCup')
cms.check('SSN')
cms.check('TimeToAct')
cms.check('BritishValues')

print("Total size in bytes in memory of Direct Method: {}".format(sys.getsizeof(hashtag_df_final)))
print("Total size in bytes in memory of CMS: {}".format(sys.getsizeof(cms)))
print("Total execution time of Direct Method: {}".format(end_time1 - start_time1))
print("Total execution time of CMS: {}".format(end_time2 - start_time2))

################## for users ##################

start_time1 = time.time()

df=pd.DataFrame(in_data)
df2=df[['user']]
df3=pd.concat([df2.drop(['user'], axis=1), df2['user'].apply(pd.Series)], axis=1)

user_df=df3.drop(df3.columns.difference(['id']), 1)
user_df['Count']=1
user_df_final=user_df.groupby(['id']).sum().sort_values(by=['Count'],ascending=False)

end_time1 = time.time()

cms=CountMinSketch(width=128,depth=8)

user_list=user_df['id'].tolist()
user_list_len=len(user_list)

user_series=pd.Series(user_list)

start_time2 = time.time()

for i in range(0,user_list_len):
    cms.add(str(user_series[i]))

end_time2 = time.time()

cms.check('2308867254')
cms.check('702125877')
cms.check('826447998')
cms.check('45715838')
cms.check('25977992')

print("Total size in bytes in memory of Direct Method: {}".format(sys.getsizeof(hashtag_df_final)))
print("Total size in bytes in memory of CMS: {}".format(sys.getsizeof(cms)))
print("Total execution time of Direct Method: {}".format(end_time1 - start_time1))
print("Total execution time of CMS: {}".format(end_time2 - start_time2))







