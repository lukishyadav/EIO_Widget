#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:11:11 2020

@author: lukishyadav

Queries Used: 

select * from darwin_rental_data where rental_started_at>'2019-01-10 00:00:01' and rental_started_at<'2020-01-10 00:00:01'


"""

import time
start=time.time()
import pandas as pd
import datetime
import statistics
from numpy import mean
import collections 
from collections import Counter as c
import pickle
from collections import Counter as C
import numpy as np

sub='data_generation/'
rental_file='Eiffel_rental_data.csv'

age_cal=1
#age_file='darwin_age.csv'
#name_file='darwin_customer_name.csv'

rentals_f=pd.read_csv(sub+rental_file)
print('rentals file read')

rentals_f.columns=['Rental ID', 'Customer ID', 'Rental Booked', 'Rental Started',
       'Rental Ended', 'Fare', 'Total to Charge',
       'Total Credits Used', 'Codes Used','not_used']


#rentals_f=rentals_f.head(500)
#rentals_f=rentals_f.head(1000)

rentals_f['Total Credits Used'].fillna(0,inplace=True)
rentals_f['Codes Used'].fillna('NA',inplace=True)

"""

REPLACING NULL RENTAL STARTED AT WITH RENTAL BOOKED AT 

"""
temp=rentals_f[rentals_f['Rental Booked'].isnull()]

for x in range(len(temp)):
    temp['Rental Booked'].iloc[x]=temp['Rental Started'].iloc[x]

rentals_f[rentals_f['Rental Booked'].isnull()]=temp
rentals_f['Rental Booked']=rentals_f['Rental Booked'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))
rentals_f['Rental Started']=rentals_f['Rental Started'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))
rentals_f['Rental Ended']=rentals_f['Rental Ended'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%d %H:%M:%S'))

rentals_f['rental_drive_duration']=rentals_f.apply(lambda x:(x['Rental Ended']-x['Rental Started']).total_seconds(),axis=1)

rentals_f['rental_duration']=rentals_f.apply(lambda x:(x['Rental Ended']-x['Rental Booked']).total_seconds(),axis=1)

rentals_f['delay']=rentals_f.apply(lambda x:(x['Rental Started']-x['Rental Booked']).total_seconds(),axis=1)

rentals_f['cid']=rentals_f['Customer ID'].copy()

rentals_f['weekday']=rentals_f['Rental Started'].apply(lambda x:x.strftime('%A'))

rentals_f['hod']=rentals_f['Rental Started'].apply(lambda x:x.hour)

bin_ranges = [0,4,8,12,16,20,24]
bin_names = ['0-4','4-8','8-12','12-16','16-20','20-24']
def bin_it(df,colname,bin_ranges,bin_names):
    df[colname+'_range'] = pd.cut(np.array(df[colname]), bins=bin_ranges,include_lowest=True)
    df[colname+'_binned'] = pd.cut(np.array(df[colname]), bins=bin_ranges,labels=bin_names,include_lowest =True)
    df[colname+'_binned'].fillna('20-24',inplace=True)
    return df[colname+'_binned']

rentals_f['hour_binned']=bin_it(rentals_f,'hod',bin_ranges,bin_names)


rentals_f = pd.concat([rentals_f,pd.get_dummies(rentals_f['hour_binned'], prefix='time_of_day')],axis=1)

rentals_f = pd.concat([rentals_f,pd.get_dummies(rentals_f['weekday'], prefix='weekday')],axis=1)

cols=list(rentals_f.columns)

D={}

for x in cols:
    if 'time_of_day' in x or "weekday" in x:
        D[x]=rentals_f.groupby('Customer ID').agg({x:'sum'}).reset_index()
        D[x].columns=['customer_id',x]


print('D calculation')

tempr=rentals_f[rentals_f['Customer ID']==10296]


duplicates=rentals_f[rentals_f['Rental ID'].duplicated(keep=False)]

# Coupon related data collection.

rentals_without_coupons=rentals_f[rentals_f['Codes Used']!='NA']

coupons_used=rentals_without_coupons.groupby(['Customer ID']).size().reset_index(name='counts')

coupon_amt_used=rentals_without_coupons.groupby(['Customer ID'])['Total Credits Used'].agg('sum').reset_index(name='coupon_amt_used')

coupon_amt_used.columns=['customer_id', 'coupon_amt_used']





tr=rentals_f.groupby(['Customer ID']).agg({'Total to Charge':'sum'}).reset_index()
tr.columns=['customer_id','total_revenue']
tr['total_revenue']=tr['total_revenue'].apply(lambda x:round(x,2))


tf=rentals_f.groupby(['Customer ID']).agg({'Fare':'sum'}).reset_index()
tf.columns=['customer_id','total_fare']
tf['total_fare']=tf['total_fare'].apply(lambda x:round(x,2))





cgroup=rentals_f.groupby('Customer ID')

atbr_drives=[]
c1=[]

def func2(x):
    #print(x.columns)
    #print(set(x['Rental ID']))
    X=x.sort_values(by=['Rental Started'])
    
    X['shiftend']=X['Rental Ended'].shift(1)
    X['avg']=X['Rental Started']-X['shiftend']
    if len(set(X['Rental ID']))==1:
        X['avg']=-1
        atbr_drives.append(-1)
        c1.append(mean(X['Customer ID']))
    else:
        
        X=X.dropna(subset=['avg']) 
        #print(len(X))
        length=len(X)
        #X['avg'].fillna(0)
        X['avg']=X['avg'].apply(lambda x:x.total_seconds())
        v=X['avg'].sum(axis = 0, skipna = True) 
        atbr_drives.append((v/length))
        c1.append(mean(X['Customer ID']))


cgroup.apply(func2)   

average_tbn_rental_drives=pd.DataFrame(np.array([c1,atbr_drives]).T,columns=['customer_id','atbr_drives'])


atbr_duration=[]
c2=[]


def func3(x):
    #print(x.columns)
    #print(set(x['Rental ID']))
    X=x.sort_values(by=['Rental Booked'])
    
    X['shiftend']=X['Rental Ended'].shift(1)
    X['avg']=X['Rental Booked']-X['shiftend']
    if len(set(X['Rental ID']))==1:
        X['avg']=-1
        atbr_duration.append(-1)
        c2.append(mean(X['Customer ID']))
    else:
        
        X=X.dropna(subset=['avg']) 
        #print(len(X))
        length=len(X)
        #X['avg'].fillna(0)
        X['avg']=X['avg'].apply(lambda x:x.total_seconds())
        v=X['avg'].sum(axis = 0, skipna = True) 
        atbr_duration.append((v/length))
        c2.append(mean(X['Customer ID']))


cgroup.apply(func3)   

average_tbn_rental_duration=pd.DataFrame(np.array([c2,atbr_duration]).T,columns=['customer_id','atbr_duration'])


print('Two Big functions')

"""
Started and Ended
"""

avg_drives=[]
c11=[]

def avg_trip_length1(x):
    avg_drives.append(sum((x['rental_drive_duration'])/len(x)))
    c11.append(mean(x['cid']))
    
cgroup.apply(avg_trip_length1)   

import numpy as np

av_drives=pd.DataFrame(np.array([c11,avg_drives]).T,columns=['customer_id','avg_rental_drives'])


"""
Booked and Ended
"""

avg_duration=[]
c22=[]

def avg_trip_length2(x):
    avg_duration.append(sum((x['rental_duration'])/len(x)))
    c22.append(mean(x['cid']))
    
cgroup.apply(avg_trip_length2)   

import numpy as np

av_duration=pd.DataFrame(np.array([c22,avg_duration]).T,columns=['customer_id','avg_rental_duration'])


"""
DAYS FROM LAST RENTAL BOOKING
"""

from datetime import datetime

v=[]
c=[]
def days_from_rental(x) :    
 max_value=max(x['Rental Booked'])
 v.append((datetime.now()-max_value).total_seconds()/(3600*24))
 c.append(mean(x['Customer ID']))
 
cgroup.apply(days_from_rental)   
 
import numpy as np

dflr=pd.DataFrame(np.array([c,v]).T,columns=['customer_id','dflr'])


df=pd.merge(tf,tr,on='customer_id',how='inner')

df=pd.merge(df,dflr,on='customer_id',how='inner')
df=pd.merge(df,av_drives,on='customer_id',how='inner')
df=pd.merge(df,av_duration,on='customer_id',how='inner')
df=pd.merge(df,average_tbn_rental_duration,on='customer_id',how='inner')
df=pd.merge(df,average_tbn_rental_drives,on='customer_id',how='inner')


abpt=rentals_f.groupby(['Customer ID']).agg({'Total to Charge':'mean'}).reset_index()
abpt.columns=['customer_id','average_bill_per_trip']
abpt['average_bill_per_trip']=abpt['average_bill_per_trip'].apply(lambda x:round(x,2))


acpt=rentals_f.groupby(['Customer ID']).agg({'Fare':'mean'}).reset_index()
acpt.columns=['customer_id','average_cost_per_trip']
acpt['average_cost_per_trip']=acpt['average_cost_per_trip'].apply(lambda x:round(x,2))

df=pd.merge(df,abpt,on='customer_id',how='inner')
df=pd.merge(df,acpt,on='customer_id',how='inner')

fd=D['weekday']

for K in list(D.keys()):
    if K != 'weekday':
        
     fd=pd.merge(fd,D[K],on='customer_id',how='inner')



df=pd.merge(fd,df,on='customer_id',how='inner')

weekday_list=['weekday_Friday', 'weekday_Monday','weekday_Thursday','weekday_Tuesday', 'weekday_Wednesday']
weekend_list=['weekday_Saturday', 'weekday_Sunday']
       
df['Weekday_rental_counts']=df.apply(lambda x:sum(x[j] for j in weekday_list),axis=1)

df['Weekend_rental_counts']=df.apply(lambda x:sum(x[j] for j in weekend_list),axis=1)

"""
CONVERTING TIME IN SECONDS TO YEAR
"""

df=pd.merge(df,coupon_amt_used,on='customer_id',how='left')
df.coupon_amt_used.fillna(0,inplace=True)
df.isnull().sum()



#Rental Counts

rental_counts=rentals_f.groupby(['Customer ID']).size().reset_index(name='counts')
rental_counts.columns=['customer_id','counts']

df=pd.merge(df,rental_counts,on='customer_id',how='inner')

#df.to_csv(sub+'final_data.csv',index=False)

df=df.drop_duplicates()


#Name of the customer


#name=pd.read_csv(sub+'NameAttempt_modified.csv')
name=pd.read_csv(sub+'eiffel_names.csv')
#name=rentals_f.groupby('Customer ID').agg({'first_name':'max','last_name':'max'}).reset_index()
name.columns=['customer_id', 'first_name', 'last_name']


df=pd.merge(df,name,on='customer_id',how='left')

df['first_name'].fillna('NF',inplace=True)
df['last_name'].fillna('LF',inplace=True)


print('Before G')

#Keeping only alphabets and spaces 

import re
df['first_name']=df['first_name'].apply(lambda x:re.sub(r'[^a-zA-Z ]+', '', x))

from gender_detector import gender_detector

detector = gender_detector.GenderDetector('us') # It can also be ar, uk, uy.
detector.guess('Premal') # => 'male'

df['gender']=df['first_name'].apply(lambda x:detector.guess(x))

#C(df['gender'])

print('After G')


unames=df[df['gender']=='unknown']


def dbr(x):
    if x ==-1:
        return x
    else:
        return x/(3600)
    
df['atbr_duration']= df['atbr_duration'].apply(dbr)
df['atbr_drives']= df['atbr_drives'].apply(dbr)

df['avg_rental_duration']=df['avg_rental_duration'].apply(dbr)
df['avg_rental_drives']=df['avg_rental_drives'].apply(dbr)

#Binning

bin_ranges = [0,50,100,150,200]
bin_names = [1,2,3,4]
df['M_range'] = pd.cut(np.array(df['total_revenue']), bins=bin_ranges,include_lowest=True)
df['M'] = pd.cut(np.array(df['total_revenue']), bins=bin_ranges,labels=bin_names,include_lowest =True)
df['M'].fillna(4,inplace=True)


R_bin_ranges = [0,50,100,150,200]
R_bin_names = [4,3,2,1]
df['R_range'] = pd.cut(np.array(df['dflr']), bins=R_bin_ranges,include_lowest=True)
df['R'] = pd.cut(np.array(df['dflr']), bins=R_bin_ranges,labels=R_bin_names,include_lowest =True)
df['R'].fillna(1,inplace=True)


F_bin_ranges = [0,50,100,150,200]
F_bin_names = [4,3,2,1]
df['F_range'] = pd.cut(np.array(df['atbr_duration']), bins=F_bin_ranges,include_lowest=True)
df['F'] = pd.cut(np.array(df['atbr_duration']), bins=F_bin_ranges,labels=F_bin_names,include_lowest =True)
df['F'].fillna(1,inplace=True)

df['R']=pd.to_numeric(df['R'])
df['M']=pd.to_numeric(df['M'])
df['F']=pd.to_numeric(df['F'])


vcheck=df[['dflr','R']]

vcheck=vcheck[vcheck['R'].isnull()]

set(df['M'])
rentals_f['paid_ride']=rentals_f['Total to Charge'].apply(lambda x:1 if x!=0 else 0)

df['age']=[0 for g in range(len(df))]

npaidrides=rentals_f.groupby('Customer ID').agg({'paid_ride':'sum',}).reset_index()
npaidrides.columns=['customer_id','No_of_paid_rides']

df=pd.merge(df,npaidrides,on='customer_id',how='left')

N=rental_file.split('_rental_data')[0]

df.columns=['Customer_id', 'weekday', 'Total_count_of_rentals_from_0-4(local time)', 'Total_count_of_rentals_from_4-8(local time)',
       'Total_count_of_rentals_from_8-12(local time)', 'Total_count_of_rentals_from_12-16(local time)', 'Total_count_of_rentals_from_16-20(local time)',
       'Total_count_of_rentals_from_20-24(local time)', 'Friday_rental_counts', 'Monday_rental_counts',
       'Saturday_rental_counts', 'Sunday_rental_counts', 'Thursday_rental_counts',
       'Tuesday_rental_counts', 'Wednesday_rental_counts', 'Total_fare(Total of all bills)', 'Total_revenue(Total of actual fees paid)',
       'Days_from_last_rental', 'Avg_rental_duration (excluding park time)', 'Avg_rental_duration (including park time)', 'Average_time_between_rental(including park time)',
       'Average_time_between_rental(Excluding park time)', 'Average_bill_per_trip', 'Average_paid_per_trip','Weekday_rental_counts', 'Weekend_rental_counts',
       'Total_Coupon_amt_used', 'Rental_Counts', 'first_name', 'last_name', 'Gender',
       'M_range', 'Monetization(range:1-4)', 'R_range', 'Recency(range:1-4)', 'F_range', 'Frequency(range:1-4)','Age','No_of_paid_rides (Rides with (Fare - Coupon value) > 0)']

df['Average_bill_per_trip'].fillna(0,inplace=True)
df['Average_paid_per_trip'].fillna(0,inplace=True)

df.drop(['first_name','last_name','R_range','F_range','M_range'],axis=1,inplace=True)


df.to_csv(N+'_the_data.csv',index=False)

print(time.time()-start)


"""
#Age Data

age=rentals_f.groupby('Customer ID').agg({'dob':'max',}).reset_index()

age.columns=['customer_id','age']

df=pd.merge(df,age,on='customer_id',how='left')

import datetime
df['age']=df['age'].apply(lambda x:datetime.datetime.strptime(x[0:19], '%Y-%m-%d') if type(x) == str else x)

df['age'].fillna(0,inplace=True)

from datetime import datetime

df['age']=df['age'].apply(lambda x:(datetime.now()-x).total_seconds()/(3600*24*365) if x!=0 else x)


#'no_of_paid_rides'


N=rental_file.split('_rental_data')[0]

df.columns=['Customer_id', 'weekday', 'Total_count_of_rentals_from_0-4(local time)', 'Total_count_of_rentals_from_4-8(local time)',
       'Total_count_of_rentals_from_8-12(local time)', 'Total_count_of_rentals_from_12-16(local time)', 'Total_count_of_rentals_from_16-20(local time)',
       'Total_count_of_rentals_from_20-24(local time)', 'Friday_rental_counts', 'Monday_rental_counts',
       'Saturday_rental_counts', 'Sunday_rental_counts', 'Thursday_rental_counts',
       'Tuesday_rental_counts', 'Wednesday_rental_counts', 'Total_fare(Total of all bills)', 'Total_revenue(Total of actual fees paid)',
       'Days_from_last_rental', 'Avg_rental_duration (excluding park time)', 'Avg_rental_duration (including park time)', 'Average_time_between_rental(including park time)',
       'Average_time_between_rental(Excluding park time)', 'Average_bill_per_trip', 'Average_paid_per_trip',
       'Total_Coupon_amt_used', 'Rental_Counts', 'first_name', 'last_name', 'Gender',
       'M_range', 'Monetization(range:1-4)', 'R_range', 'Recency(range:1-4)', 'F_range', 'Frequency(range:1-4)', 'Age','No_of_paid_rides']

df['Average_bill_per_trip'].fillna(0,inplace=True)
df['Average_paid_per_trip'].fillna(0,inplace=True)

df.drop(['first_name','last_name'],axis=1,inplace=True)
df.to_csv('generated_data/'+N+'_the_data.csv',index=False)

print(time.time()-start)



#df['check']=df.apply(lambda x:1 if x['Rental_Counts']>=x['No_of_paid_rides'] else 0,axis=1)

"""