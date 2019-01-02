# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:41:46 2017

@author: elwi
"""
import pandas as pd
import datetime as dt
import numpy as np

"""PATH = BSS DATA, PATH2 = BSS STATION DATA, PATH3 = POSTAL CODE DATA """
path = 'C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\city_bike_stats_Combined_Full_Season.csv'
path2 = 'C:\HY-Data\ELWI\BikeSharingData\Mikkos_Desciptive_Excel\StationCoordinates.csv'

df1= pd.read_csv(path, sep = ";", encoding = "utf8")
StatCoordinates = pd.read_csv(path2, sep=';')

type(df1)
print (df1.columns)
print(df1.head())
print(StatCoordinates.head())

pd.options.display.float_format = '{:.2f}'.format # format precision in describe output

df1.describe()

# TRIP DISTANCE FILTER (filter out less than 100m and more than 70km trips
df_filtered = df1.ix[(df1['covered_distance'] > 100) & (df1['covered_distance'] != 0) &
                     (df1['covered_distance'] <= 70000)]  
                     
# TRIP DURATION FILTER (filter out less than 60 and more than 18000 (5h) seconds trips)
df_filtered = df_filtered.ix[(df1['duration'] > 60) & (df1['duration'] != 0) &
                     (df1['duration'] <= 18000)]
                     
# FORMULA FILTER (filter out all the VIP-users and other users without proper formula)                     
df_filtered = df_filtered.ix[(df1['hsl_formula'] == "Year") | (df1['hsl_formula'] == "Week") |
                     (df1['hsl_formula'] == "Day")] 
                     
# STATION FILTER (IF NEEDED) (first separate the station code and change the data type then filter out the stations outside Helsinki)                                      
df_filtered['dep_stat_code'] = df_filtered['departure_station'].str[:3]
df_filtered['ret_stat_code'] = df_filtered['return_station'].str[:3]

df_filtered['dep_stat_code'] = df_filtered.dep_stat_code.astype(float)
df_filtered['ret_stat_code'] = df_filtered.ret_stat_code.astype(float)

df_filtered = df_filtered.ix[(df_filtered['departure_station1'] >= 1) & (df_filtered['departure_station1'] < 151)]
df_filtered = df_filtered.ix[(df_filtered['return_station1'] >= 1) & (df_filtered['return_station1'] < 151)]

# CONVERT THE DEPARTURE AND RETURN COLUMNS AS THE DATETIME TYPE, CHOOSE DEPARTURE AND THE INDEX AND CALCULATE WEEKDAY TO A NEW COLUMN 
df_filtered['departure_time'] = pd.to_datetime(df_filtered['departure_time'], infer_datetime_format=True, format='%m%d%Y %H%M%S')
df_filtered['return_time'] = pd.to_datetime(df_filtered['return_time'], infer_datetime_format=True, format='%m%d%Y %H%M%S')

df_filtered.datetimeindex = df_filtered['departure_time']

df_filtered['weekday'] = df_filtered['departure_time'].dt.weekday

# CONVERT THE BIRTHDAY COLUMN TO THE DATETIME TYPE, THEN CORRECT ALL THE VALUES WHERE THE DATE IS < NOW (E.G YEAR 2067)
# FINALLY CALCULATE A NEW COLUMN FOR THE AGE OF THE PERSON         
now = pd.Timestamp(dt.datetime.now())
df_filtered['hsl_birthday'] = pd.to_datetime(df_filtered['hsl_birthday'].astype(str), errors = "coerce")
df_filtered['hsl_birthday'] = df_filtered['hsl_birthday'].where(df_filtered['hsl_birthday'] < now, df_filtered['hsl_birthday'] -  np.timedelta64(100, 'Y'))   
df_filtered['hsl_age'] =  (now - df_filtered['hsl_birthday']).astype('timedelta64[Y]')

# CALCULATE A SPEED COLUMN (UNIT KM/H) 
df_filtered["speed"] = (df_filtered["covered_distance"] / 1000) / (df_filtered["duration"]/3600)

# TRIP SPEED FILTER (filter out faster than 40km/h trips
df_filtered = df_filtered.ix[(df_filtered['speed'] < 40)]

# REORDER THE COLUMNS
df_filtered = df_filtered[['departure_time', 'return_time', 'account', 'departure_station1',
       'departure_station2', 'return_station1', 'return_station2', 'formula',
       'covered_distance', 'duration', 'speed','weekday','id', 'uid', 'hsl_formula',
        'hsl_birthday', 'hsl_gender', 'hsl_age', 'hsl_postal_code', 'hsl_region', 'hsl_city', 'hsl_country']]

# REPLACE SPECIAL CHARACTERS
df_filtered.replace(['Ã¤', 'Ã„' , 'Ã¶' , 'Ã–' , 'Ã¥', 'Ã…' ], ['ä', 'Ä', 'ö', 'Ö', 'å', 'Å'])

# TAKE A SAMPLE
sample = df_filtered.sample(100)    
sample.head(10)        

# FOR CHECKING THE DATA               
df_filtered.describe()
df_filtered.dtypes
df_filtered.head(10)
df_filtered.columns

# WRITE THE CLEANED DATASET OUT AS CSV FORMAT
output_fp = "C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS2017_Full_season_Cleaned.csv"
df_filtered.to_csv(output_fp, sep=',')