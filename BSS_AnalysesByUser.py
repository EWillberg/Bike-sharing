# -*- coding: utf-8 -*-
"""

Usage:
    This script is intended for analyzing bike-sharing data provided by the Helsinki region transport (HSL) and
    CityBikeFinland

    The script contains several functions for bike-sharing data analysis and their execution

Created:
    Fri Oct 20 15:41:46 2017

Author:
    Elias Willberg


"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from scipy.stats import pearsonr
from joblib import delayed, Parallel
import time
import scipy


pd.options.display.float_format = '{:.6g}'.format # format precision in describe output

############################################ FUNCTION PART ###########################################################

# Implementation of the following functions:
# ... 1) roundTripCalculator - Function to calculate whether the trip starts where the earlier trip has ended
# (or nearby, within 500m) on the same day):
# ... 2) roundTripCalculator2 - Parallelized test version of the function 1, not faster though
# ... 3) roundTripDaysByUser - Function to calculate the number of days when the user's last trip has
# returned where his first trip of the day started
# ... 4) roundTripDaysByUserUsingNearStation - Function to calculate the number of days when the user's last trip
# has returned NEARBY where his first trip of the day started
# ... 5) resetIndexIfNecessary -  Function to reset the index if necessary 
# ... 6) nearStation - Function to check has the departure station been nearby(within 500m along a network
# of the given return station
# ... 7) isItTheSameDay - Function to check has the departure and return happened on the same day
# ... 8) numberOfTripDays - Function to calculate the number of trip days per user and the trip/day ratio by user
# ... 9) weekWeekendRatio - Function to calculate the number of weekday trips and weekend trips and the absolute
# and the relative ratio between weekday/weekend use
# ... 10) timeDifference - Function to check the time how long some other function is running

### FUNCTION 1 (SEE DESCRIPTION ABOVE)
      
def roundTripCalculator(dataframe, newColumnName, uidColumn, depTimeColumn, depStationColumn,retTimeColumn, 
                         retStationColumn, optimalRoutesDataframe, optimalRouteColumn):
    StartTime = time.time()                         
    sameStationCount, nearStationCount = 0,0                        
    dataframe[newColumnName] = np.nan
    dataframe.sort_values([uidColumn, depTimeColumn], ascending=[True,True,], inplace = True)
    resetIndexIfNecessary(dataframe)    
    
    for index, row in dataframe.iterrows():
        print(index)               
        if index == 0:
            dataframe.at[index, newColumnName] = 0
            previousRow = row                
        else:
            if row[uidColumn] == previousRow[uidColumn]:
                if (row[depStationColumn] == previousRow[retStationColumn]) & (isItTheSameDay(row[depTimeColumn], previousRow[retTimeColumn])): # if checking round trips
                    dataframe.at[index, newColumnName] = 1
                    sameStationCount += 1
                elif (nearStation(optimalRoutesDataframe, optimalRouteColumn, depStationColumn, 
                                 row[depStationColumn], retStationColumn, previousRow[retStationColumn])) & (isItTheSameDay(row[depTimeColumn], previousRow[retTimeColumn])):
                    dataframe.at[index, newColumnName] = 1
                    nearStationCount += 1
                else:
                    dataframe.at[index, newColumnName] = 0                   
                previousRow = row                   
            else:
                dataframe.at[index, newColumnName] = 0 
                previousRow = row                           
    
    EndTime = time.time()            
    print(sameStationCount, nearStationCount)
    print(timeDifference(StartTime, EndTime))            
    return dataframe 
    

""" ### FUNCTION 2 (SEE DESCRIPTION ABOVE)
def roundTripCalculator2(dataframe, newColumnName, uidColumn, depTimeColumn, depStationColumn,retTimeColumn, 
                         retStationColumn, optimalRoutesDataframe, optimalRouteColumn):                     
    dataframe[newColumnName] = np.nan
    dataframe.sort_values([uidColumn, depTimeColumn], ascending=[True,True,], inplace = True)
    resetIndexIfNecessary(dataframe)    
    #returnInfoTuple = dataframe.loc[0,[retTimeColumn, depTimeColumn]]
    userID = dataframe.iloc[0][uidColumn]  
        
    with Parallel(n_jobs=4,
              backend="threading",
              verbose=100) as parallel:              
             
         delayedFunc = []
         previousRow = None
         for index, row in dataframe.iterrows():
             delayedFunc.append(delayed(roundTripCalculatorByRow)(dataframe,row, previousRow, index, newColumnName,uidColumn,
                    userID,depTimeColumn, 
                    depStationColumn, retTimeColumn, retStationColumn, 
                    optimalRoutesDataframe, optimalRouteColumn))
             previousRow = row
           
         returns = parallel(delayedFunc)                    
                           
   # print(sameStationCount, nearStationCount)
    return dataframe  
        

def roundTripCalculatorByRow(dataframe,row, previousRow, index, newColumnName, uidColumn,userID, depTimeColumn,
                             depStationColumn, retTimeColumn, retStationColumn, optimalRoutesDataframe, optimalRouteColumn):
    if index == 0:
        dataframe.at[index, newColumnName] = 0
        userID = row[uidColumn]
        #returnInfoTuple = dataframe.loc[0,[retStationColumn, retTimeColumn]]
        #returnInfoTuple[0] = row[retStationColumn] 
        #returnInfoTuple[1] = row[retTimeColumn]   
    else: 
        # previousRow = dataframe.iloc[index-1]
        #returnInfoTuple = previousRow.loc[[retStationColumn, retTimeColumn, uidColumn] #dataframe.loc[index-1,[retStationColumn, retTimeColumn]]
        previousUserID = previousRow[uidColumn] #dataframe.loc[index-1,[uidColumn, retTimeColumn]]
        if row[uidColumn] == previousUserID:
            if (row[depStationColumn] == previousRow[retStationColumn]) & (isItTheSameDay(row[depTimeColumn], previousRow[retTimeColumn])): # if checking round trips
                dataframe.at[index, newColumnName] = 1
                #sameStationCount += 1
            elif (nearStation(optimalRoutesDataframe, optimalRouteColumn, depStationColumn, 
                             row[depStationColumn], retStationColumn, previousRow[retStationColumn])) & (isItTheSameDay(row[depTimeColumn], previousRow[retTimeColumn])):
                dataframe.at[index, newColumnName] = 1
                #nearStationCount += 1
            else:
                dataframe.at[index, newColumnName] = 0        
         #   returnInfoTuple[0] = row[retStationColumn] 
        #    returnInfoTuple[1] = row[retTimeColumn] 
               
        else:
            dataframe.at[index, newColumnName] = 0   
         #   returnInfoTuple[0] = row[retStationColumn] 
        #    returnInfoTuple[1] = row[retTimeColumn] 
          #  userID = row[uidColumn]  
                             
"""
### FUNCTION 3 (SEE THE DESCRIPTION ABOVE)
def roundTripDaysByUser (dataframe, uidColumn, depTimeColumn,depStationColumn,retStationColumn, dayOfTheYearColumn):
    users_round_trip_ratios = {}                       
    dataframe.sort_values([uidColumn, depTimeColumn], ascending=[True,True,], inplace = True)
    first = dataframe.groupby([uidColumn, dayOfTheYearColumn]).first().reset_index()
    last = dataframe.groupby([uidColumn, dayOfTheYearColumn]).last().reset_index()
    dayCount= 0
    roundTripDayCount = 0
    
    for indx, row in first.iterrows():
        print(indx, " / ", len(first)-1) 
        
        if indx == 0:
            dayCount += 1           
            if row[depStationColumn] == last[retStationColumn].iloc[indx]:
                roundTripDayCount += 1
                
        elif indx == (len(first)-1):
            dayCount += 1            
            if row[depStationColumn] == last[retStationColumn].iloc[indx]:
                roundTripDayCount += 1            
            if roundTripDayCount != 0:
                ratio = roundTripDayCount/dayCount             
            else:
                ratio  = 0                 
            users_round_trip_ratios[first[uidColumn].iloc[indx]] = ratio   
            
        elif row[uidColumn] == first[uidColumn].iloc[indx-1]:
            dayCount += 1            
            if row[depStationColumn] == last[retStationColumn].iloc[indx]:
                roundTripDayCount += 1
        
        else:
            if roundTripDayCount != 0:
                ratio = roundTripDayCount/dayCount 
            else:
                ratio  = 0            
            users_round_trip_ratios[first[uidColumn].iloc[indx-1]] = ratio 
            dayCount, roundTripDayCount = 0,0

    return pd.DataFrame.from_dict(users_round_trip_ratios, orient='index').reset_index()           

### FUNCTION 4 (SEE DESCRIPTION ABOVE) 
def roundTripDaysByUserUsingNearStation(dataframe, uidColumn, depTimeColumn, depStationColumn,retStationColumn, dayOfTheYearColumn,
                                         optimalRoutesDataframe,optimalRouteColumn):
    users_round_trip_ratios = {}                         
    dataframe.sort_values([uidColumn, depTimeColumn], ascending=[True,True,], inplace = True)
    first = dataframe.groupby([uidColumn, dayOfTheYearColumn]).first().reset_index()
    last = dataframe.groupby([uidColumn, dayOfTheYearColumn]).last().reset_index()
    dayCount= 0
    roundTripDayCount = 0
    
    for indx, row in first.iterrows():
        print(indx, " / ", len(first)-1)  
        
        if indx == 0:
            dayCount += 1           
            if row[depStationColumn] == last[retStationColumn].iloc[indx]:
                roundTripDayCount += 1
            elif nearStation(optimalRoutesDataframe, optimalRouteColumn, depStationColumn, row[depStationColumn], 
                             retStationColumn, last[retStationColumn].iloc[indx]):
                roundTripDayCount += 1
                 
        elif indx == (len(first) -1):
            dayCount += 1            
            if row[depStationColumn] == last[retStationColumn].iloc[indx]:
                roundTripDayCount += 1
            elif nearStation(optimalRoutesDataframe, optimalRouteColumn, depStationColumn, row[depStationColumn], 
                             retStationColumn, last[retStationColumn].iloc[indx]):    
                roundTripDayCount += 1                 
            if roundTripDayCount != 0:
                ratio = roundTripDayCount/dayCount             
            else:
                ratio  = 0                 
            users_round_trip_ratios[first[uidColumn].iloc[indx]] = ratio   
            
        elif row[uidColumn] == first[uidColumn].iloc[indx-1]:
            dayCount += 1            
            if row[depStationColumn] == last[retStationColumn].iloc[indx]:
                roundTripDayCount += 1
            elif nearStation(optimalRoutesDataframe, optimalRouteColumn, depStationColumn, row[depStationColumn], 
                             retStationColumn, last[retStationColumn].iloc[indx]):
                roundTripDayCount += 1                 
        else:
            if roundTripDayCount != 0:
                ratio = roundTripDayCount/dayCount 
            else:
                ratio  = 0            
            users_round_trip_ratios[first[uidColumn].iloc[indx-1]] = ratio 
            dayCount, roundTripDayCount = 0,0

    return pd.DataFrame.from_dict(users_round_trip_ratios, orient='index').reset_index()               
    
### FUNCTION 5 (SEE DESCRIPTION ABOVE) 
def resetIndexIfNecessary (dataframe):
    if dataframe.index[0] != 0:
        dataframe.reset_index(inplace = True) 
        
### FUNCTION 6 (SEE DESCRIPTION ABOVE)                 
def nearStation (optimalRoutesDF, distanceColumn, depStationColumn,depStatID, retStationColumn, retStatID):
    r =  optimalRoutesDF.loc[(optimalRoutesDF[depStationColumn] == depStatID) & (optimalRoutesDF[retStationColumn] == retStatID)]    
    if (r[distanceColumn].values[0] < 500) & (r[distanceColumn].values[0] != 0): # If the nearest neighboring station is within 500m but not the station itself
        return True
    else:
        return False
        
### FUNCTION 7 (SEE DESCRIPTION ABOVE)         
def isItTheSameDay (depTime, retTime):
     if pd.to_datetime(depTime).dayofyear == pd.to_datetime(retTime).dayofyear:
         return True
     else:
         return False
         
### FUNCTION 8 (SEE DESCRIPTION ABOVE)      
def numberOfTripDays (dataframe, uidColumn, dayOfTheYearColumn):    
    totalDays = 175 # total number of trip days in 2017 minus the number of months (error in the dataset as the last day of each month is missing)
    userDaysDF = dataframe.groupby([uidColumn, dayOfTheYearColumn]).first()      
    userDaysCount = userDaysDF.groupby(level=[0]).size()
    userDayRatio = userDaysCount / totalDays
    
    numberOfTripDaysDF = pd.concat([userDaysCount, userDayRatio], axis=1)
    
    return numberOfTripDaysDF  
    
### FUNCTION 9 (SEE DESCRIPTION ABOVE)   
def weekVsWeekendRatio (dataframe, idColumn, uidColumn, weekdayColumn):
    
    weekTripsDF = dataframe[(dataframe[weekdayColumn] <= 4)].groupby([uidColumn, weekdayColumn]).count()  
    weekTripCount = weekTripsDF[idColumn].groupby(level=[0]).sum()
    weekTripCount.rename('weekCount', inplace  = True)
    
    weekendTripsDF = dataframe[(dataframe[weekdayColumn] >= 5)].groupby([uidColumn, weekdayColumn]).count()  
    weekendTripCount = weekendTripsDF[idColumn].groupby(level=[0]).sum()
    weekendTripCount.rename('weekendCount', inplace  = True)       
    
    weekOrWeekendDF= pd.concat([weekTripCount, weekendTripCount],axis=1).fillna(0)
    weekOrWeekendDF['absRatio'] = weekOrWeekendDF.apply(lambda row: row['weekCount'] if (row['weekendCount'] == 0) else (row['weekendCount'] if (row['weekCount'] == 0) else row['weekCount'] / row['weekendCount']), axis =1)
    weekOrWeekendDF['relaRatio'] = weekOrWeekendDF.apply(lambda row: row['weekCount']/5  if (row['weekendCount'] == 0) else (row['weekendCount'] /2 if (row['weekCount'] == 0) else (row['weekCount']/5) / (row['weekendCount']/2)), axis = 1)
   
    return weekOrWeekendDF.reset_index()
         
### FUNCTION 10 (SEE DESCRIPTION ABOVE)          
def timeDifference(startTime, endTime):
    totalTime = (endTime - startTime) /60  #min
    return totalTime

############################################ FUNCTION PART ENDS  ###########################################################
           
            
############################################ THE MAIN PART 1 - DATA MANIPULATION ###################################################
         
BSS_path = 'C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS2017_Full_season_with_DistanceDifference_1_49m.csv'
Postal_Areas_path  = 'Z:\Gradu\Data\Shapefiles_and_Other_GIS_files\Postal_areas_inside_BSS_Area.csv'
PT_near_BSS_stations_path = 'Z:\Gradu\Data\Shapefiles_and_Other_GIS_files\PT_Near_BSS_Stations.csv'
Postal_codes_path = 'C:\HY-Data\ELWI\Postinumeroalueet\Postal_codes_Finland.csv'
Optimal_routes_path = 'C:\HY-Data\ELWI\BikeSharingData\Processed2017\Station_Pair_Routing\BSS_Network_Routes.csv'
ManipulatedDF_path = 'C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS_Full_season_1_34m_Loops_RoundTrips_Included.csv'
ManipulatedAllUsersDF_path = 'C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS_Full_season_1_49m_Loops_RoundTrips_Included.csv'
Network_drive_BSS_data_path = 'Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS2017_Full_season_with_DistanceDifference.csv'

df= pd.read_csv(BSS_path, sep = ",", encoding  ="utf8");
df1 = pd.read_csv(Network_drive_BSS_data_path, sep = ",", encoding  ="utf8");
Postal_areas= pd.read_csv(Postal_Areas_path, sep = ",", encoding  ="utf8");
PT_near_BSS_stations = pd.read_csv(PT_near_BSS_stations_path, sep = "," , encoding  = "utf8")
postalCodes= pd.read_csv(Postal_codes_path, sep=';', encoding = "utf8")
optimalRoutes = pd.read_csv(Optimal_routes_path, sep=',')
            
df.head(10)
optimalRoutes.rename(index = str, columns={"Stat_ID_1": "departure_station1","Stat_ID" :"return_station1"}, inplace=True)

# Take a copy of all users dataframe because those users with "formula" == week/day will be deleted when running some of the following filterings
# ... At the end of this section, there should be ~1,49 million users in the "allUsersDF" and 1,34 million users in "df"
allUsersDF = df.copy()

# Join postal code information (municipality name & city) and replace respective columns (NOTE: THIS OPERATION DELETES MOST DAY AND WEEK USERS)
df=df.merge(postalCodes[['nimi', 'kuntanimi']], 
            left_on='hsl_postal_code', right_on=postalCodes['postinro'])
columns = ['hsl_city','hsl_region' ]
df = df.drop(columns, axis=1)       

df = df.rename(columns={'nimi': 'hsl_region', 'kuntanimi': 'hsl_city'})

# A new column to indicate whether a user lives inside or outside the BSS coverage area
Listed_postal_areas= Postal_areas['Posno'].tolist()
df['InsideArea'] = df['hsl_postal_code'].apply(lambda a: 1 if int(a) in (Listed_postal_areas) else 0)

# Two new columns to indicate whether the departure or return station have been near public transport hubs
PT_near_station_IDs = PT_near_BSS_stations['Stat_ID'].tolist()

df['PT_dep'] = df['departure_station1'].apply(lambda a: 1 if int(a) in (PT_near_station_IDs) else 0)
df['PT_ret'] = df['return_station1'].apply(lambda a: 1 if int(a) in (PT_near_station_IDs) else 0)
df['PT_trip'] = df.apply(lambda row: 1 if (row['PT_dep'] == 1) else 1 if (row['PT_ret'] == 1) else 0, axis = 1)

    #Do the same for all users dataframe
allUsersDF['PT_dep'] = allUsersDF['departure_station1'].apply(lambda a: 1 if int(a) in (PT_near_station_IDs) else 0)
allUsersDF['PT_ret'] = allUsersDF['return_station1'].apply(lambda a: 1 if int(a) in (PT_near_station_IDs) else 0)
allUsersDF['PT_trip'] = allUsersDF.apply(lambda row: 1 if (row['PT_dep'] == 1) else 1 if (row['PT_ret'] == 1) else 0, axis = 1)

#  # A new column to indicate whether the trip has been taken during weekdays or weekends
df['WeekOrWeekend'] = df['weekday'].apply(lambda a: 1 if a <= 4 else 0)

    #Do the same for all users dataframe
allUsersDF['WeekOrWeekend'] = allUsersDF['weekday'].apply(lambda a: 1 if a <= 4 else 0)

# New columns to indicate the hour of departure and return, the day of the year, and the number of month for the trips
df['DepHour'] = pd.DatetimeIndex(df['departure_time']).hour
df['RetHour'] = pd.DatetimeIndex(df['return_time']).hour
df['DayOfTheYear'] = pd.DatetimeIndex(df['departure_time']).dayofyear
df['Month'] = pd.DatetimeIndex(df['departure_time']).month
    
    #Do the same for all users dataframe
allUsersDF['DepHour'] = pd.DatetimeIndex(allUsersDF['departure_time']).hour
allUsersDF['RetHour'] = pd.DatetimeIndex(allUsersDF['return_time']).hour
allUsersDF['DayOfTheYear'] = pd.DatetimeIndex(allUsersDF['departure_time']).dayofyear
allUsersDF['Month'] = pd.DatetimeIndex(allUsersDF['departure_time']).month

# Calculate the standard deviation of 1) station usage & 2) departure hours for each user (Step 1: Calculate a frequency table, step 2: calculate the STD of station usage for each user from the frequency table, Step 3: merge the values to the whole dataframe)   
DepStationFrequencyValues = df['id'].groupby([df["uid"], df["departure_station1"]]).count()
DepStationFrequencyTable = DepStationFrequencyValues.to_frame()
DepStationFrequencyTable = DepStationFrequencyTable.unstack(level=-1)
DepStationFrequencyTable['DepStat_STD_'] = DepStationFrequencyTable.std(axis = 1, skipna = True) # standard deviation of usage per user among the stations the user has used
DepStationFrequencyTable.fillna(0, inplace = True)
DepStationFrequencyTable['DepStat_STD_ALL'] = DepStationFrequencyTable.std(axis = 1, skipna = False) # standard deviation of usage per user of all the stations
DepStationFrequencyTable.reset_index(inplace = True)

DepHourFrequencyValues = df['id'].groupby([df["uid"], df["DepHour"]]).count()
DepHourFrequencyTable = DepHourFrequencyValues.to_frame()
DepHourFrequencyTable = DepHourFrequencyTable.unstack(level=-1)
DepHourFrequencyTable['DepHour_STD_'] = DepHourFrequencyTable.std(axis = 1, skipna = True) # standard deviation of usage per user among the departure hours the user has started
DepHourFrequencyTable.fillna(0, inplace = True)
DepHourFrequencyTable['DepHour_STD_ALL'] = DepHourFrequencyTable.std(axis = 1, skipna = False) # standard deviation of usage per user of all the possible departure hours
DepHourFrequencyTable.reset_index(inplace = True)

df = pd.merge(df, DepStationFrequencyTable[['DepStat_STD_','DepStat_STD_ALL']], left_on = df['uid'], right_on = DepStationFrequencyTable['uid', ""])
df = pd.merge(df, DepHourFrequencyTable[['DepHour_STD_','DepHour_STD_ALL']], left_on = df['uid'], right_on = DepHourFrequencyTable['uid', ""])

    #Do the same for all users dataframe
DepStationFrequencyValues = allUsersDF['id'].groupby([allUsersDF["uid"], allUsersDF["departure_station1"]]).count()
DepStationFrequencyTable = DepStationFrequencyValues.to_frame()
DepStationFrequencyTable = DepStationFrequencyTable.unstack(level=-1)
DepStationFrequencyTable['DepStat_STD_'] = DepStationFrequencyTable.std(axis = 1, skipna = True) # standard deviation of usage per user among the stations the user has used
DepStationFrequencyTable.fillna(0, inplace = True)
DepStationFrequencyTable['DepStat_STD_ALL'] = DepStationFrequencyTable.std(axis = 1, skipna = False) # standard deviation of usage per user of all the stations
DepStationFrequencyTable.reset_index(inplace = True)

allUsersDF = pd.merge(allUsersDF, DepStationFrequencyTable[['DepStat_STD_','DepStat_STD_ALL']], left_on = allUsersDF['uid'], right_on = DepStationFrequencyTable['uid', ""])

# Calculate how many trips have been 1) round trips  
optimalRoutes.loc[(optimalRoutes['Total_Leng'] < 500) & (optimalRoutes['Total_Leng'] > 10)].count() #test how many station has its nearest neighbor station within 500m

# Calculate columns to indicate whether the trip has been a part of travel chain
testSet1 = df.iloc[0:10000]         
testSet = df.iloc[0:100000]

testSet1 = roundTripCalculator(testSet1, 'nearDepStartFromRet', 'uid', 'departure_time','departure_station1', 'return_time', 'return_station1', optimalRoutes, 'Total_Leng')

df = roundTripCalculator(df, 'depStartFromRet', 'uid', 'departure_time','departure_station1', 'return_time', 'return_station1', optimalRoutes, 'Total_Leng') # disable elif from roundTripCalculator function
df = roundTripCalculator(df, 'nearDepStartFromRet', 'uid', 'departure_time','departure_station1', 'return_time', 'return_station1', optimalRoutes, 'Total_Leng') # enable elif from roundTripCalculator function
            
allUsersDF = roundTripCalculator(allUsersDF, 'depStartFromRet', 'uid', 'departure_time','departure_station1', 'return_time', 'return_station1', optimalRoutes, 'Total_Leng')     
allUsersDF = roundTripCalculator(allUsersDF, 'nearDepStartFromRet', 'uid', 'departure_time','departure_station1', 'return_time', 'return_station1', optimalRoutes, 'Total_Leng')

# Calculate whether a trip has been a loop
testSet1['loop'] = np.where(testSet1['departure_station1'] == testSet1['return_station1'],1, 0)

df['loop'] = np.where(df['departure_station1'] == df['return_station1'],1, 0) # disable elif from the roundTripCalculator function

allUsersDF['loop'] = np.where(allUsersDF['departure_station1'] == allUsersDF['return_station1'],1, 0) # enable elif from the roundTripCalculator function

# TAKE OUTPUTS BEFORE THE NEXT STAGE 
df.to_csv("C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS_Full_season_1_34m_Loops_RoundTrips_Included.csv", sep=',', index  =False)
allUsersDF.to_csv("C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS_Full_season_1_49m_Loops_RoundTrips_Included.csv", sep=',', index  =False)    

############################################ THE MAIN PART 1 - DATA MANIPULATION ENDS ###########################################################

############################################ THE MAIN PART 2 - DATA AGGREGATION   ################################################################

#Use the manipulated dataframes if necessary
manipulatedDF= pd.read_csv(ManipulatedDF_path, sep = ",", encoding  ="utf8");
manipulatedAllUsersDF= pd.read_csv(ManipulatedAllUsersDF_path, sep = ",", encoding  ="utf8");
manipulatedDF.to_csv("C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS_Full_season_1_34m_Cleaned_and_Preprocessed.csv", sep=',', index  =False)

# manipulatedDF.drop(["Unnamed: 0",'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis  = 1, inplace = True)
# manipulatedDF['PT_trip'] = manipulatedDF.apply(lambda row: 1 if (row['PT_dep'] == 1) else 1 if (row['PT_ret'] == 1) else 0, axis = 1)

testSet1 = manipulatedDF.iloc[0:11000]                
                                                              
# Columns to be aggregated  (The two last columns aggregate the number of departures and returns that occurred near public tranport hubs)
aggregations = {
         'id' : 'count',       
        'covered_distance' : ['mean','median'],
        'duration' : ['mean','median'],        
  #      'hsl_age' : 'mean',        
        'speed' : ['mean','median'],  
        'diff' : ['mean','median']  ,
#        'InsideArea': 'mean', 
        'departure_station1' : 'nunique',
        'return_station1' : 'nunique',
        'DepHour' : 'nunique',
        'RetHour' : 'nunique',
        'DayOfTheYear' : 'nunique',
        'Month' : 'nunique',
        'PT_dep' : lambda x: x[x == 1].count(),
        'PT_ret'  : lambda x: x[x == 1].count(),
        'PT_trip' : lambda x: x[x == 1].count(), 
        'depStartFromRet' :lambda x: x[x == 1].count(),
        'nearDepStartFromRet' :lambda x: x[x == 1].count(),        
        'loop' :lambda x: x[x == 1].count(),
        ('DepStat_STD_', "") : lambda x: x.value_counts().index[0],        
 #       '(\'DepStat_STD_ALL\', \'\')': lambda x: x.value_counts().index[0],
 #       '(\'DepHour_STD_\', \'\')' : lambda x: x.value_counts().index[0],
 #       '(\'DepHour_STD_ALL\', \'\')' : lambda x: x.value_counts().index[0],
        }

# Grouping phase (need to do with merging as groupby function excludes all the missing values from the gender column)        
UserGroup = manipulatedDF.groupby(['uid', 'hsl_postal_code', 'hsl_region', 'hsl_city', 'hsl_country']).agg(aggregations).reset_index()
UserGroupWithGender = manipulatedDF.groupby(['uid', 'hsl_gender', 'hsl_postal_code', 'hsl_region', 'hsl_city', 'hsl_country']).agg(aggregations).reset_index()
CombinedUserGroup = pd.merge(UserGroup, UserGroupWithGender['hsl_gender', ""].to_frame(), how ='outer', left_on='uid', right_on=UserGroupWithGender['uid',""])
CombinedUserGroup.fillna("N/A")

CombinedUserGroup.describe()

AllUsersCombinedUserGroup = manipulatedAllUsersDF.groupby(['uid','formula']).agg(aggregations).reset_index()
AllUsersCombinedUserGroup.fillna("N/A")
    
# Calculate further columns to the user table                              
CombinedUserGroup['tripsPerDay'] = CombinedUserGroup['id']['count'] / 175    # total number of trip days in 2017 minus the number of months (error in the dataset as the last day of each month is missing)
CombinedUserGroup['dep_PT_Pros'] = (CombinedUserGroup['PT_dep']['<lambda>'] / CombinedUserGroup['id']['count']) # Percentage of Public transport hubs departures
CombinedUserGroup['ret_PT_Pros'] = (CombinedUserGroup['PT_ret']['<lambda>'] / CombinedUserGroup['id']['count']) # Percentage of Public transport hubs returns
CombinedUserGroup['PT_trip_Pros'] = (CombinedUserGroup['PT_trip']['<lambda>'] / CombinedUserGroup['id']['count'])

CombinedUserGroup['dep_Ret_Ratio'] = (CombinedUserGroup['departure_station1']['nunique'] / CombinedUserGroup['return_station1']['nunique']) # Ratio between the number of unique departure stations and the number of unique return stations
CombinedUserGroup['dep_Ret_Hour_Ratio'] = (CombinedUserGroup['DepHour']['nunique'] / CombinedUserGroup['RetHour']['nunique']) # Ratio between the number of unique departure stations and the number of unique return stations

CombinedUserGroup['loop_Ratio'] = (CombinedUserGroup['loop']['<lambda>'] / CombinedUserGroup['id']['count']) 
CombinedUserGroup['depStartFromRet_Ratio'] = (CombinedUserGroup['depStartFromRet']['<lambda>'] / CombinedUserGroup['id']['count']) 
CombinedUserGroup['nearDepStartFromRet_Ratio'] = (CombinedUserGroup['nearDepStartFromRet']['<lambda>'] / CombinedUserGroup['id']['count']) 
    
    
AllUsersCombinedUserGroup['tripsPerDay'] = AllUsersCombinedUserGroup['id']['count'] / 175    # total number of trip days in 2017 minus the number of months (error in the dataset as the last day of each month is missing)
AllUsersCombinedUserGroup['dep_PT_Pros'] = (AllUsersCombinedUserGroup['PT_dep']['<lambda>'] / AllUsersCombinedUserGroup['id']['count']) # Percentage of Public transport hubs departures
AllUsersCombinedUserGroup['ret_PT_Pros'] = (AllUsersCombinedUserGroup['PT_ret']['<lambda>'] / AllUsersCombinedUserGroup['id']['count']) # Percentage of Public transport hubs returns
AllUsersCombinedUserGroup['PT_trip_Pros'] = (AllUsersCombinedUserGroup['PT_trip']['<lambda>'] / AllUsersCombinedUserGroup['id']['count'])

AllUsersCombinedUserGroup['dep_Ret_Ratio'] = (AllUsersCombinedUserGroup['departure_station1']['nunique'] / AllUsersCombinedUserGroup['return_station1']['nunique']) # Ratio between the number of unique departure stations and the number of unique return stations
AllUsersCombinedUserGroup['dep_Ret_Hour_Ratio'] = (AllUsersCombinedUserGroup['DepHour']['nunique'] / AllUsersCombinedUserGroup['RetHour']['nunique']) # Ratio between the number of unique departure stations and the number of unique return stations

AllUsersCombinedUserGroup['loop_Ratio'] = (AllUsersCombinedUserGroup['loop']['<lambda>'] / AllUsersCombinedUserGroup['id']['count']) 
AllUsersCombinedUserGroup['depStartFromRet_Ratio'] = (AllUsersCombinedUserGroup['depStartFromRet']['<lambda>'] / AllUsersCombinedUserGroup['id']['count']) 
AllUsersCombinedUserGroup['nearDepStartFromRet_Ratio'] = (AllUsersCombinedUserGroup['nearDepStartFromRet']['<lambda>'] / AllUsersCombinedUserGroup['id']['count']) 

# Calculate the ratio for each user in how many days their last trip of the day has ended where the day's first one started (or nearby). 
# ... After that, merge the ratio to CombinedUserGroup dataframe
RoundTripDF = roundTripDaysByUser(manipulatedDF, "uid", "departure_time", "departure_station1", "return_station1", 'DayOfTheYear')
CopyRoundTripDF = RoundTripDF.copy()

RoundTripUsingNearStationDF = roundTripDaysByUserUsingNearStation(manipulatedDF, "uid", "departure_time", "departure_station1", "return_station1", 'DayOfTheYear',
                                       optimalRoutes, 'Total_Leng')
                                   
RoundTripDF.columns = pd.MultiIndex.from_product([RoundTripDF.columns, ['']]) # Create an additional index level to match the levels of CombinedUserGroup dataframe
RoundTripUsingNearStationDF.columns = pd.MultiIndex.from_product([RoundTripUsingNearStationDF.columns, ['']]) # Create an additional index level to match the levels of CombinedUserGroup dataframe

RoundTripDF.rename(columns = {'index' :'uid'}, inplace = True)
RoundTripUsingNearStationDF.rename(columns = {'index' :'uid'}, inplace = True)

CombinedUserGroup = CombinedUserGroup.merge(RoundTripDF, how='left', left_on = 'uid', right_on = 'uid')
CombinedUserGroup = CombinedUserGroup.merge(RoundTripUsingNearStationDF, how='left', left_on ='uid', right_on = 'uid')    
    #Do the same for all users dataframe
AllUsersRoundTripDF = roundTripDaysByUser(manipulatedAllUsersDF, "uid", "departure_time", "departure_station1", "return_station1", 'DayOfTheYear')
AllUsersRoundTripDF.columns = pd.MultiIndex.from_product([AllUsersRoundTripDF.columns, ['']]) # Create an additional index level to match the levels of CombinedUserGroup dataframe
AllUsersRoundTripDF.rename(columns = {'index' :'uid'}, inplace = True)
AllUsersCombinedUserGroup = AllUsersCombinedUserGroup.merge(AllUsersRoundTripDF, how='left', left_on = 'uid', right_on = 'uid')

# Calculating the number of trips days
TripDaysDF = numberOfTripDays(manipulatedDF, 'uid', 'DayOfTheYear').reset_index()
TripDaysDF.columns = pd.MultiIndex.from_product([TripDaysDF.columns, ['']])
CombinedUserGroup = CombinedUserGroup.merge(TripDaysDF, how='left', left_on ='uid', right_on = 'uid')   

   #Do the same for all users dataframe
AllUsersTripDaysDF = numberOfTripDays(manipulatedAllUsersDF, 'uid', 'DayOfTheYear').reset_index()
AllUsersCombinedUserGroup = AllUsersCombinedUserGroup.merge(AllUsersTripDaysDF, how='left', left_on ='uid', right_on = 'uid')

# Calculating the week/weekend counts and ratios for each user
Week_WeekendDF = weekVsWeekendRatio(manipulatedDF,'id', 'uid','weekday')
CombinedUserGroup = CombinedUserGroup.merge(Week_WeekendDF, how='left', left_on ='uid', right_on = 'uid')
   #Do the same for all users dataframe
AllUsersWeek_WeekendDF = weekVsWeekendRatio(manipulatedAllUsersDF,'id', 'uid','weekday')
AllUsersCombinedUserGroup = AllUsersCombinedUserGroup.merge(AllUsersWeek_WeekendDF, how='left', left_on ='uid', right_on = 'uid')

# Rename CombinedUserGroup fields 
CombinedUserGroupCopy = CombinedUserGroup.copy()
CombinedUserGroup= CombinedUserGroupCopy

CombinedUserGroup.drop("uid", axis  = 1, inplace = True)
CombinedUserGroup.columns = ["uid", "hsl_postal_code", "hsl_region", "hsl_city", "hsl_country", "departure_station1_nunique","loopCount","DayOfTheYear_nunique",
                            "return_station1_nunique", "depStartFromRet_count", "PT_ret_count","speed_mean", "speed_median","insideArea", "nearDepStartFromRet_count",
                            "RetHour_nunique", "hsl_age", "DepStatSTD", "duration_mean", "duration_median", "PT_dep_count", "DepHour_nunique","trip_count", 
                            "distance_mean", "distance_median", "DepStatSTD_All", "month_nunique","DepHourSTD_All", "diff_mean", "diff_median","DepHourSTD",                                                       
                             "PT_trip_count",  "hsl_gender","tripsPerDay", "dep_PT_pros", "ret_PT_pros", "PT_trip_pros", "dep_Ret_count_ratio", "dep_Ret_Hour_ratio", 
                             "loop_ratio", "depStartFromRet_ratio", "nearDepStartFromRet_ratio","Days_RetToStartDep_ratio", "Days_NearRetToStartDep_ratio", 
                             "userDayCount", "userDayRatio", "weekdayTripCount", "weekendTripCount", "week_weekend_absRatio", "week_weekend_relaRatio"]

AllUsersCombinedUserGroup.drop("uid", axis  = 1, inplace = True)
AllUsersCombinedUserGroup.columns = ["uid","formula","RetHour_nunique", "PT_trip_count","nearDepStartFromRet_count","speed_mean", "speed_median",
                            "depStartFromRet_count", "DayOfTheYear_nunique", "PT_ret_count","duration_mean", "duration_median","distance_mean", "distance_median", 
                            "trip_count","DepStatSTD", "loopCount",  "diff_mean", "diff_median","month_nunique","departure_station1_nunique",
                            "return_station1_nunique","PT_dep_count","DepHour_nunique","tripsPerDay", "dep_PT_pros", "ret_PT_pros", "PT_trip_pros",
                            "dep_Ret_count_ratio","dep_Ret_Hour_ratio","loop_ratio","depStartFromRet_ratio", "nearDepStartFromRet_ratio",
                            "userDayCount", "userDayRatio", "weekdayTripCount", "weekendTripCount",  "week_weekend_absRatio", "week_weekend_relaRatio"]                           
                             
cols = CombinedUserGroup.columns.tolist()                  
 
cols = ['uid', 'hsl_age','hsl_gender','hsl_postal_code', 'hsl_region', 'hsl_city', 'hsl_country', 'insideArea', 'trip_count', 
'departure_station1_nunique', 'return_station1_nunique', 'dep_Ret_count_ratio','duration_mean', 'duration_median','speed_mean', 'speed_median', 
'distance_mean', 'distance_median','DayOfTheYear_nunique', 'month_nunique','diff_mean', 'diff_median','loopCount', 'DepHour_nunique', 
'RetHour_nunique', 'dep_Ret_Hour_ratio', 'DepStatSTD', 'DepStatSTD_All',  'DepHourSTD', 'DepHourSTD_All', 
'PT_dep_count', 'PT_ret_count', 'PT_trip_count','dep_PT_pros','ret_PT_pros','PT_trip_pros','depStartFromRet_count', 'nearDepStartFromRet_count', 
'depStartFromRet_ratio', 'nearDepStartFromRet_ratio', 'Days_RetToStartDep_ratio', 'Days_NearRetToStartDep_ratio','loop_ratio',
 'userDayCount', 'tripsPerDay','userDayRatio', 'weekdayTripCount', 'weekendTripCount', 'week_weekend_absRatio', 'week_weekend_relaRatio']   

AllUserscols = ['uid',"formula", 'trip_count', 'departure_station1_nunique', 'return_station1_nunique', 'dep_Ret_count_ratio','duration_mean', 
'duration_median','speed_mean', 'speed_median', 'distance_mean', 'distance_median','DayOfTheYear_nunique', 'month_nunique','diff_mean', 
'diff_median','loopCount', 'DepHour_nunique', 'RetHour_nunique', 'dep_Ret_Hour_ratio', 'DepStatSTD', 'PT_dep_count', 'PT_ret_count', 
'PT_trip_count','dep_PT_pros','ret_PT_pros','PT_trip_pros','depStartFromRet_count', 'nearDepStartFromRet_count', 'depStartFromRet_ratio',
'nearDepStartFromRet_ratio', 'loop_ratio','userDayCount', 'tripsPerDay','userDayRatio', 'weekdayTripCount', 'weekendTripCount', 
'week_weekend_absRatio', 'week_weekend_relaRatio']   

    
CombinedUserGroup = CombinedUserGroup[cols]
AllUsersCombinedUserGroup = AllUsersCombinedUserGroup[AllUserscols]

UserTableOutputPath = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_UsersNewVersion.csv"
AllUserTableOutputPath = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_ALLUsersNewVersion.csv"

CombinedUserGroup.to_csv(UserTableOutputPath, sep=',', index = False)
AllUsersCombinedUserGroup.to_csv(AllUserTableOutputPath, sep=',', index = False)

############################################ THE MAIN PART 2 - DATA AGGREGATION ENDS ###############################################################

############################################ THE MAIN PART 3 - DATA ANALYSIS  ######################################################################

CombinedUserGroup_path = 'Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_UsersNewVersion.csv'
AllUsersCombinedUserGroup_path = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_ALLUsersNewVersion.csv"

CombinedUserGroup = pd.read_csv(CombinedUserGroup_path, sep = ",", encoding  ="utf8");
AllUsersCombinedUserGroup = pd.read_csv(AllUsersCombinedUserGroup_path, sep = ",", encoding  ="utf8");

# Manipulating data for gender analyses
pd.value_counts(CombinedUserGroup['hsl_gender'].values, sort=False)
males = CombinedUserGroup.loc[CombinedUserGroup['hsl_gender'] == "male"]   
females = CombinedUserGroup.loc[CombinedUserGroup['hsl_gender'] == "female"]   

MalesMedian = males.median()
FemalesMedian = females.median()

CombinedUserGroup['age_dec'] = CombinedUserGroup.hsl_age.map(lambda hsl_age:  10 * (hsl_age // 10))  #create the age by decade column
GenderMedian = pd.concat([MalesMedian, FemalesMedian],axis = 1)
GenderAgeMeans= pd.DataFrame(CombinedUserGroup.groupby(["hsl_gender","age_dec"], axis = 0).mean())
GenderAgeMeans= np.transpose(GenderAgeMeans)

CombinedUserGroup['genderNumeric'] = CombinedUserGroup.apply(lambda row: 1 if (row['hsl_gender'] == 'male') else 2 if (row['hsl_gender'] == 'female') else (0), axis =1)

# Descriptive statistics d
CombinedUserGroup.describe()
males.describe()
females.describe()

# Manipulating and analyzing data for insider/outsider checks
insiders = CombinedUserGroup.loc[CombinedUserGroup['insideArea'] == 1]   
outsiders = CombinedUserGroup.loc[CombinedUserGroup['insideArea'] == 0]   

insideAreaTrips= df.loc[df1['InsideArea'] == 1]
outsideAreaTrips = df.loc[df1['InsideArea'] == 0]

insideAreaWeekdayTrips = df.loc[(df['InsideArea'] == 1) & (df['weekday'] < 5)]
outsideAreaWeekdayTrips = df.loc[(df['InsideArea'] == 0) & (df['weekday'] < 5)]

InsidersMedian = insiders.median()
InsidersMean = insiders.mean()
InsidersSTD = insiders.std()

OutsidersMedian = outsiders.median()
OutsidersMean = outsiders.mean()
OutsidersSTD= outsiders.std()

CombinedMedian = pd.concat([OutsidersMedian, InsidersMedian,],axis = 1)
CombinedMean = pd.concat([OutsidersMean, InsidersMean,],axis = 1)
CombinedSTD = pd.concat([OutsidersSTD, InsidersSTD,],axis = 1)

topStationUsersInside= pd.value_counts(insideAreaTrips['departure_station2'].values, sort= True).reset_index()
topStationUsersOutside = pd.value_counts(outsideAreaTrips['departure_station2'].values, sort= True).reset_index()

InOutStationDF = pd.DataFrame(topStationUsersInside).merge(topStationUsersOutside, on='index', how ="left")

# Subscription type analyzes
AllUsersCombinedUserGroup['tripsPerUserDay'] = AllUsersCombinedUserGroup.apply(lambda row: row['trip_count'] if (row['formula'] == 'Day') else (row['trip_count'] / 7 if (row['formula'] == 'Week') else (row['trip_count'] / 177)), axis =1)
AllUsersCombinedUserGroup['NumericFormula'] = AllUsersCombinedUserGroup.apply(lambda row: 1 if (row['formula'] == 'Day') else 2 if (row['formula'] == 'Week') else (3), axis =1)

dayUsers = AllUsersCombinedUserGroup.loc[AllUsersCombinedUserGroup['formula'] == "Day"]   
weekUsers = AllUsersCombinedUserGroup.loc[AllUsersCombinedUserGroup['formula'] == "Week"]   
yearUsers = AllUsersCombinedUserGroup.loc[AllUsersCombinedUserGroup['formula'] == "Year"]   

dayUsersMedian = dayUsers.median()
weekUsersMedian = weekUsers.median()
yearUsersMedian = yearUsers.median()

CombinedSubscriptionMedian = pd.concat([dayUsersMedian, weekUsersMedian,yearUsersMedian],axis = 1)

# Activity of use analyzes
AllUsersCombinedUserGroup.sort_values('trip_count', ascending=False, inplace = True)
CombinedUserGroup.sort_values('trip_count', ascending=False, inplace = True)

q = pd.qcut(AllUsersCombinedUserGroup["trip_count"],5, labels=["1","2","3","4","5"])
q2 = pd.qcut(CombinedUserGroup["trip_count"],5, labels=["1","2","3","4","5"])

AllUsersCombinedUserGroup['ActivityQ'] = q
AllUsersCombinedUserGroup['ActivityQ']= AllUsersCombinedUserGroup['ActivityQ'].astype(int)

CombinedUserGroup['ActivityQ'] = q2
CombinedUserGroup['ActivityQ']= CombinedUserGroup['ActivityQ'].astype(int)

ActivityQ1 = AllUsersCombinedUserGroup.loc[(AllUsersCombinedUserGroup['ActivityQ'] == "1")]
ActivityQ2 = AllUsersCombinedUserGroup.loc[(AllUsersCombinedUserGroup['ActivityQ'] == "2")]
ActivityQ3 = AllUsersCombinedUserGroup.loc[(AllUsersCombinedUserGroup['ActivityQ'] == "3")]
ActivityQ4 = AllUsersCombinedUserGroup.loc[(AllUsersCombinedUserGroup['ActivityQ'] == "4")]
ActivityQ5 = AllUsersCombinedUserGroup.loc[(AllUsersCombinedUserGroup['ActivityQ'] == "5")]

ActivityGroupsMedian= AllUsersCombinedUserGroup.groupby(['ActivityQ']).median()
ActivityGroupsCount= AllUsersCombinedUserGroup.groupby(['ActivityQ']).count()

# Manipulating and analyzing data for age comparisons
CombinedUserGroup = CombinedUserGroup.loc[(CombinedUserGroup['hsl_age'] >10) & (CombinedUserGroup['hsl_age'] <80)]
CombinedUserGroup['age_dec'] = CombinedUserGroup.hsl_age.map(lambda hsl_age:  10 * (hsl_age // 10))  #create the age by decade column

# Manipulating and analyzing data for shortest path VS realized path comparisons
diffUnder100 = CombinedUserGroup.loc[CombinedUserGroup['diff_median'] < 100] .mean()
diff100_299 = CombinedUserGroup.loc[(CombinedUserGroup['diff_median'] >= 100) & (CombinedUserGroup['diff_median']<299)].mean() 
diff300_499 = CombinedUserGroup.loc[(CombinedUserGroup['diff_median'] >=300) & (CombinedUserGroup['diff_median']<499)].mean()
diffOver500 = CombinedUserGroup.loc[(CombinedUserGroup['diff_median'] >= 500)].mean() 

diffMeans = pd.concat([diffUnder100,diff100_299,diff300_499, diffOver500],axis = 1)

# Checking the departures from the public transport hubs at different times of the day
aggregationsByHour = {
         'id' : 'count',
         'PT_dep' : lambda x: x[x == 1].count(),
         'PT_ret': lambda x: x[x == 1].count()
         }

HourlyStats = manipulatedDF.groupby(['DepHour']).agg(aggregationsByHour).reset_index() 
HourlyStats['Dep_PT_Pros'] = HourlyStats['PT_dep'] / HourlyStats['id']
HourlyStats['Ret_PT_Pros'] = HourlyStats['PT_ret'] / HourlyStats['id']

Weekdays = df.loc[df['weekday'] < 5]
HourlyStatsWeekdays = Weekdays.groupby(['DepHour']).agg(aggregationsByHour).reset_index() 
HourlyStatsWeekdays['Dep_PT_Pros'] = HourlyStatsWeekdays['PT_dep'] / HourlyStatsWeekdays['id']
HourlyStatsWeekdays['Ret_PT_Pros'] = HourlyStatsWeekdays['PT_ret'] / HourlyStatsWeekdays['id']

# Getting user's most frequent postal areas 1) by trip 2) by user
TopPostalAreasByTrip = pd.value_counts(manipulatedDF['hsl_region'].values, sort= True)
TopPostalAreasByUser = pd.value_counts(CombinedUserGroup['hsl_region'].values, sort= True)

valueAggregations = {
       "id" : "count",
       "uid" : "nunique", 
       "departure_station1" : lambda x: x.value_counts(dropna=False).index[0],
       "departure_station2" : lambda x: x.value_counts(dropna=False).index[0],
       "return_station1"  : lambda x: x.value_counts(dropna=False).index[0],
       "return_station2" : lambda x: x.value_counts(dropna=False).index[0]       
    }
    
CombinedPostalAreasDF = manipulatedDF.groupby(["hsl_postal_code", "hsl_region", "hsl_city"]).agg(valueAggregations).reset_index()

CombinedPostalAreasDF = CombinedPostalAreasDF.rename(columns= {"id":"id_count", 
                                                                   "uid":"uid_nunique", 
                                                                   "departure_station1":"depStat_mode", 
                                                                   "departure_station2":"depStat2_mode",
                                                                   "return_station1":"retstat_mode",
                                                                   "return_station2":"retStat2_mode"})
                                                                   
CombinedPostalAreasDF["hsl_postal_code"] = CombinedPostalAreasDF["hsl_postal_code"].astype(int)
CombinedPostalAreasDF["hsl_postal_code"] = CombinedPostalAreasDF["hsl_postal_code"].astype(str)
                                                   
CombinedPostalAreasDF["hsl_postal_code"] = CombinedPostalAreasDF["hsl_postal_code"].apply(lambda x: str("00"+ x) if (len(x) == 3) else (str("0"+ x) if (len(x) == 4) else (str(x))))                                                           

type(CombinedPostalAreasDF["hsl_postal_code"][0])


# Checking the most popular stations
valueAggregationsStations = {
       "id" : "count",
       "uid" : "nunique",
    }

DepStationCountsDF = manipulatedAllUsersDF.groupby(['departure_station1', 'departure_station2']).agg(valueAggregationsStations).reset_index()
RetStationCountsDF = manipulatedAllUsersDF.groupby(['return_station1', 'return_station2']).agg(valueAggregationsStations).reset_index()

#Correlation checks
SubsetDF = CombinedUserGroup[['hsl_age', 'departure_station1_nunique','trip_count','duration_median','speed_median','distance_median','diff_median','DepStatSTD',
                             'DepHourSTD', 'PT_trip_pros','depStartFromRet_ratio','Days_RetToStartDep_ratio','tripsPerDay','week_weekend_relaRatio']]
                             
SubsetDF['hsl_gender'] = SubsetDF['hsl_gender'].apply(lambda row: 1 if (row == "male") else (2 if (row == "female") else np.nan))
SubsetDF['hsl_city'] = SubsetDF['hsl_city'].apply(lambda row: 1 if (row == "Helsinki") else (2 if (row == "Espoo") else (3 if (row == "Vantaa") else 4)))

corr = SubsetDF.corr()
g = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap=sns.diverging_palette(220, 10, as_cmap=True))
g.set_yticklabels(g.get_yticklabels(), rotation =0)
g.set_xticklabels(g.get_yticklabels(), rotation =90)

corr.style.background_gradient()

scipy.stats.spearmanr(SubsetDF['insideArea'], SubsetDF['diff_median'])

#Temporal checks
manipulatedDF['PT_trip'] = manipulatedDF.apply(lambda row: 1 if (row['PT_dep'] == 1) else 1 if (row['PT_ret'] == 1) else 0, axis = 1)

weekdayMornings = manipulatedDF.loc[(manipulatedDF['DepHour']>= 7) & (manipulatedDF['DepHour']<= 9) & (manipulatedDF['WeekOrWeekend'] == 1)]
weekdayAfternoons = manipulatedDF.loc[(manipulatedDF['DepHour']>= 15) & (manipulatedDF['DepHour']<= 17) & (manipulatedDF['WeekOrWeekend'] == 1)]
weekdayMorningsOutsiders = manipulatedDF.loc[(manipulatedDF['DepHour']>=7) & (manipulatedDF['DepHour']<= 9) & (manipulatedDF['WeekOrWeekend'] == 1) & (manipulatedDF['InsideArea'] == 0)]
weekdayMorningsInsiders = manipulatedDF.loc[(manipulatedDF['DepHour']>= 7) & (manipulatedDF['DepHour']<= 9) & (manipulatedDF['WeekOrWeekend'] == 1) & (manipulatedDF['InsideArea'] == 1)]
weekdayAfternoonsOutsiders = manipulatedDF.loc[(manipulatedDF['DepHour']>= 15) & (manipulatedDF['DepHour']<= 17) & (manipulatedDF['WeekOrWeekend'] == 1) & (manipulatedDF['InsideArea'] == 0)]
weekdayAfternoonsInsiders = manipulatedDF.loc[(manipulatedDF['DepHour']>= 15) & (manipulatedDF['DepHour']<= 17) & (manipulatedDF['WeekOrWeekend'] == 1) & (manipulatedDF['InsideArea'] == 1)]

weekendsMornings = manipulatedDF.loc[(manipulatedDF['DepHour']>= 7) & (manipulatedDF['DepHour']<= 9) & (manipulatedDF['WeekOrWeekend'] == 0)]
weekendsAfternoons = manipulatedDF.loc[(manipulatedDF['DepHour']>= 15) & (manipulatedDF['DepHour']<= 17) & (manipulatedDF['WeekOrWeekend'] == 0)]

weekendMorningsOutsiders = manipulatedDF.loc[(manipulatedDF['DepHour']>= 7) & (manipulatedDF['DepHour']<= 9) & (manipulatedDF['WeekOrWeekend'] == 0) & (manipulatedDF['InsideArea'] == 0)]
weekendMorningsInsiders = manipulatedDF.loc[(manipulatedDF['DepHour']>=7) & (manipulatedDF['DepHour']<= 9) & (manipulatedDF['WeekOrWeekend'] == 0) & (manipulatedDF['InsideArea'] == 1)]

weekendAfternoonsOutsiders = manipulatedDF.loc[(manipulatedDF['DepHour']>= 15) & (manipulatedDF['DepHour']<= 17) & (manipulatedDF['WeekOrWeekend'] == 0) & (manipulatedDF['InsideArea'] == 0)]
weekendAfternoonsInsiders = manipulatedDF.loc[(manipulatedDF['DepHour']>=15) & (manipulatedDF['DepHour']<= 17) & (manipulatedDF['WeekOrWeekend'] == 0) & (manipulatedDF['InsideArea'] == 1)]

# Outputting to CSV and Excel
UserTableOutputPath = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_UsersNewVersion.csv"
CombinedUserGroup.to_csv(UserTableOutputPath, sep=',', index = False)

AllUsersTableOutputPath = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_AllUsersNewVersion.csv"
AllUsersCombinedUserGroup.to_csv(AllUsersTableOutputPath, sep=',', index = False)

PostalAreaDFOutputPath = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_PostalAreaCounts.csv"
CombinedPostalAreasDF.to_csv(PostalAreaDFOutputPath, sep=',', index = False)

DepStationTableOutputPath = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_StationCounts.csv"
RetStationTableOutputPath = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_RetStationCounts.csv"

DepStationCountsDF.to_csv(DepStationTableOutputPath, sep=',', index = False)
RetStationCountsDF.to_csv(RetStationTableOutputPath, sep=',', index = False)

SubsetDFOutputPath = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_Users_OnlyCorrelatingColumns.csv"
SubsetDF.to_csv(SubsetDFOutputPath, sep=',', index = False)

InOutStationDFPath = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_StationCount_InidersOutsiders.csv"
InOutStationDF.to_csv(InOutStationDFPath, sep=',', index = False)

writer = pd.ExcelWriter('Z:\Gradu\Data\Excel\BSS_Full_season_Users.xlsx')
CombinedUserGroup.to_excel(writer, 'UserInfo')
writer.save()

sample = df.sample(100)

