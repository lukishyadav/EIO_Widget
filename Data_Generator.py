#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:54:45 2020

@author: lukishyadav
"""

import pandas as pd
from collections import Counter
from datetime import datetime
from pyproj import Proj
from math import sqrt
import numpy as np
from math import radians, cos, sin, asin, sqrt

fname='wave3.csv'

fname='daytona_rental_data.csv'
fname='eiffel_rental_data.csv'
df=pd.read_csv(fname)


min(df['rental_started_at'])


max(df['rental_started_at'])

df10=df.head(10)
df10.dtypes

#f.drop(['r_id','RID'],axis=1,inplace=True)

df.isnull().sum()



df.dropna(subset=['end_lat','end_long','start_lat','start_long','rental_started_at','rental_booked_at'],inplace=True)



#df['object_data-credit_amount_used'].fillna(0,inplace=True)


dt_columns=['rental_started_at','rental_booked_at', 'rental_ended_at']


for col in dt_columns:
    df[col] = df[col].apply(lambda x:datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))

"""

from settings import region
from datetime import datetime




selected_region = region['oakland']
REGION_TIMEZONE = selected_region['timezone']


# converts incoming data to proper timezone
def convert_datetime_columns(df, columns):
    for col in columns:
        try:
            df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert(REGION_TIMEZONE)
        except TypeError:
            df[col] = df[col].dt.tz_convert(
                   'UTC').dt.tz_convert(REGION_TIMEZONE)




df['rental_started_at'].iloc[0]


dt_columns=['rental_started_at','rental_booked_at', 'rental_ended_at']



df['credit_amount_used'].fillna(0,inplace=True)

df.isnull().sum()

df.dropna(inplace=True)


for col in dt_columns:
    df[col] = df[col].apply(lambda x:datetime.strptime(x[0:19], '%Y-%m-%dT%H:%M:%S'))

fd=df.copy()

convert_datetime_columns(fd, dt_columns)

fd['rental_started_at'].iloc[0]

"""


len(set(df['vehicle_id']))


d=dict(Counter(df['vehicle_id']))

sorted_d=sorted(d.items(), key=lambda x: x[1], reverse=True)






df=df.sort_values(by=['rental_started_at'])

"""
df141=df[df['vehicle_id']==141]

df141s=df141.shift(-1)
df141s.dropna(inplace=True)
df141asof=pd.merge_asof(df141[['rental_booked_at','rental_started_at', 'rental_ended_at','end_lat', 'end_long', 'start_lat', 'start_long']], df141[['rental_started_at','end_lat', 'end_long', 'start_lat', 'start_long']], left_on='rental_ended_at',right_on='rental_started_at', direction='forward')      
"""


vehicle_dict={}



df['vehicleid']=df['vehicle_id'].copy()
vehicle_group=df.groupby('vehicleid')


cons_columns=['vehicle_id','rental_booked_at_x', 'rental_started_at_x', 'rental_ended_at_x',
       'end_lat_x', 'end_long_x', 'start_lat_x', 'start_long_x',
       'rental_booked_at_y', 'rental_started_at_y', 'rental_ended_at_y',
       'end_lat_y', 'end_long_y', 'start_lat_y', 'start_long_y']

global master_df
vehicle_dict['master_df']=pd.DataFrame(data=[],columns=['vehicle_id','rental_booked_at_x', 'rental_started_at_x', 'rental_ended_at_x',
       'end_lat_x', 'end_long_x', 'start_lat_x', 'start_long_x',
       'rental_booked_at_y', 'rental_started_at_y', 'rental_ended_at_y',
       'end_lat_y', 'end_long_y', 'start_lat_y', 'start_long_y'])


def vehicle_func(x):
    Xs=x.sort_values(by=['rental_ended_at'])
    Xe=x.sort_values(by=['rental_started_at'])
    vid=max(x['vehicle_id'])
    print(vid)
    #print(x.isnull().sum())
    #print(len(Xs))
    if len(Xs)>1:
      vehicle_dict[vid]=pd.merge_asof(Xs[['vehicle_id','rental_booked_at','rental_started_at', 'rental_ended_at','end_lat', 'end_long', 'start_lat', 'start_long']], Xe[['rental_booked_at','rental_started_at', 'rental_ended_at','end_lat', 'end_long', 'start_lat', 'start_long']], left_on='rental_ended_at',right_on='rental_started_at', direction='forward',allow_exact_matches=False) 
    else:
     print(vid,'2nd condition')  
     vehicle_dict[vid]=pd.DataFrame(np.array([np.nan for x in range(len(cons_columns))]).reshape(1,15),columns=cons_columns)
         
     #vehicle_dict[vid]=x[['vehicle_id','rental_booked_at','rental_started_at', 'rental_ended_at','end_lat', 'end_long', 'start_lat', 'start_long']]  
    vehicle_dict['master_df']=pd.concat([vehicle_dict['master_df'],vehicle_dict[vid]]) 
 
    
    
vehicle_group.apply(vehicle_func)


master_df=vehicle_dict['master_df']

"""

dftest=df[df['vehicle_id']==219]
dftest=dftest.sort_values(by=['rental_started_at'])

dftest2=dftest.sort_values(by=['rental_ended_at'])

dftest141asof=pd.merge_asof(dftest[['rental_booked_at','rental_started_at', 'rental_ended_at','end_lat', 'end_long', 'start_lat', 'start_long']], dftest2[['rental_started_at','end_lat', 'end_long', 'start_lat', 'start_long']], left_on='rental_ended_at',right_on='rental_started_at', direction='forward')      


for x in range(100):
    x=x*10
    dft=dftest[x:x+10]
    dftest141asof=pd.merge_asof(dft[['rental_booked_at','rental_started_at', 'rental_ended_at','end_lat', 'end_long', 'start_lat', 'start_long']], dft[['rental_started_at','end_lat', 'end_long', 'start_lat', 'start_long']], left_on='rental_ended_at',right_on='rental_started_at', direction='forward')      


dfff=dft.iloc[[2,3],:]

#dftest.reset_index(inplace=True)



#dftest=dftest.reset_index()
for x in list(dftest.index):
    if dftest['rental_started_at'].iloc[x]==dftest['rental_started_at'].iloc[x+1]:
        print('WTF',x,x+1)
      
        
observe=dftest.loc[[92,93],:]  
        
        
for x in range(len(df)):
    if df['rental_started_at'].iloc[x]==df['rental_started_at'].iloc[x+1]:
        print('WTF',x,x+1)
        
observe=df.loc[[146578,146579],:]     


check =vehicle_dict[2]  
"""        


master_df.isnull().sum() 
master_df.dropna(inplace=True)       

cm=master_df[['end_lat_x', 'end_lat_y', 'end_long_x',
       'end_long_y', 'rental_booked_at_x',
       'rental_booked_at_y','rental_ended_at_x',
       'rental_ended_at_y', 'rental_started_at_x',
       'rental_started_at_y',  'start_lat_x', 'start_lat_y', 'start_long_x', 'start_long_y']]
      
cm.isnull().sum()
  
def convert_to_mercator(lngs, lats):
    projection = Proj(init='epsg:3857')
    xs = []
    ys = []
    for lng, lat in zip(lngs, lats):
        x, y = projection(lng, lat)
        xs.append(x)
        ys.append(y)
    return xs, ys

cl=['end_lat_x', 'end_long_x', 'start_lat_x', 'start_long_x','end_lat_y', 'end_long_y', 'start_lat_y', 'start_long_y']


for lcol in list(range(0,len(cl),2)):    
    master_df['mrc_'+cl[lcol+1]],master_df['mrc_'+cl[lcol]]=convert_to_mercator(master_df[cl[lcol+1]], master_df[cl[lcol]])


#RADIUS CHECK       
    
    
def distance(x):
    a = x['mrc_end_lat_x'] - x['mrc_start_lat_y']
    b = x['mrc_end_long_x'] - x['mrc_start_long_y']
    c = sqrt(a * a  +  b * b)
    
    """
    if (c < radius):
            print("inside")
    else:
            print("outside")
    """
    return c        
        

master_df['distance']=master_df.apply(distance,axis=1)


def haversine(x):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [x['end_long_x'], x['end_lat_x'], x['start_long_y'], x['start_lat_y']])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r*1000


master_df['haversine_distance']=master_df.apply(haversine,axis=1)

def within_radius(x):
    if x['haversine_distance']<50:
        return 1
    else:
        return 0
    

master_df['within_radius']=master_df.apply(within_radius,axis=1)        


master_df.to_csv('generated_data/'+fname.split('.csv')[0]+'generated.csv',index=False)

Counter(master_df['within_radius'])