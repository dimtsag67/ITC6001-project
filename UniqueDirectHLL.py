# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 22:35:55 2020

@author: Teoi
"""

import pandas as pd
import os
import psutil
process = psutil.Process(os.getpid())
import time
import hyperloglog
import sys
import json
from more_itertools import sliced

in_data=[]

for root, subdirs, files in os.walk('C:/Users/Teoi/Desktop/json files'):
    for file in files:
        with open (file,errors='ignore',encoding="UTF-8") as infile:
            for line in infile:
                in_data.append(json.loads(line))

################## for hashtags ##################

df=pd.DataFrame(in_data)
df2=df[['entities']]
df3=pd.concat([df2.drop(['entities'], axis=1), df2['entities'].apply(pd.Series)], axis=1)

hashtag_df=df3.explode('hashtags')
hashtag_df.drop(hashtag_df.columns.difference(['hashtags']), 1, inplace=True)
hashtag_df=pd.concat([hashtag_df.drop(['hashtags'], axis=1), hashtag_df['hashtags'].apply(pd.Series)], axis=1)
hashtag_df.drop(hashtag_df.columns.difference(['text']), 1, inplace=True)
hash_list=hashtag_df['text'].tolist()
hash_series=pd.Series(hash_list)

rows_number=len(hash_series)

x=[*(sliced(([*range(0,rows_number)]),1000))]    #slice the df by 1000 lines counter


start_time1 = time.time()

counter=[]
for lst in x:
    counter=counter+lst
    unique_hashtags_df=hash_series.iloc[counter].unique()
unique_hashtags=len(unique_hashtags_df)

end_time1 = time.time()


start_time2 = time.time()

hash_series=hash_series.fillna(0)

hll_hashtags = hyperloglog.HyperLogLog(0.01)  # accept 1% counting error

for list in x:                                     #iteration through the lists in thee counter to get each 1000 lines
    for item in list:                              #iteration to the item in each list which represents each row number form the data dataframe
        if hash_series.iloc[item] !=0:
            hll_hashtags.add(hash_series.iloc[item])

unique_hashtags_hyper=len(hll_hashtags)

end_time2 = time.time()

print("For Unique Hashtags:")
print("Total size in bytes in memory of Direct Method: {}".format(sys.getsizeof(unique_hashtags_df)))
print("Total size in bytes in memory of HLL: {}".format(sys.getsizeof(hll_hashtags)))
print("Total execution time of Direct Method: {}".format(end_time1 - start_time1))
print("Total execution time of HLL: {}".format(end_time2 - start_time2))
print("Unique hashtags of Direct Method: {}".format(unique_hashtags))
print("Unique hashtags of HLL: {}".format(unique_hashtags_hyper))

################## for users ##################

df=pd.DataFrame(in_data)
df2=df[['user']]
df3=pd.concat([df2.drop(['user'], axis=1), df2['user'].apply(pd.Series)], axis=1)

user_df=df3.drop(df3.columns.difference(['id']), 1)
user_list=user_df['id'].tolist()
user_series=pd.Series(user_list)

rows_number=len(user_series)

x=[*(sliced(([*range(0,rows_number)]),1000))]    #slice the df by 1000 lines counter

start_time1 = time.time()

counter=[]
for lst in x:
    counter=counter+lst
    unique_users_df=user_series.iloc[counter].unique()
unique_users=len(unique_users_df)

end_time1 = time.time()

start_time2 = time.time()

users_series=user_series.fillna(0)

hll_users = hyperloglog.HyperLogLog(0.01)  # accept 1% counting error

for list in x:                                     #iteration through the lists in thee counter to get each 1000 lines
    for item in list:                              #iteration to the item in each list which represents each row number form the data dataframe
        if users_series.iloc[item] !=0:
            hll_users.add(users_series.iloc[item])

unique_users_hyper=len(hll_users)

end_time2 = time.time()

print("For Unique Users:")
print("Total size in bytes in memory of Direct Method: {}".format(sys.getsizeof(unique_users_df)))
print("Total size in bytes in memory of HLL: {}".format(sys.getsizeof()))
print("Total execution time of Direct Method: {}".format(end_time1 - start_time1))
print("Total execution time of HLL: {}".format(end_time2 - start_time2))
print("Unique hashtags of Direct Method: {}".format(unique_users))
print("Unique hashtags of HLL: {}".format(unique_users_hyper))




