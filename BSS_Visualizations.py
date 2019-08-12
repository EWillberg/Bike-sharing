# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:05:10 2018

Usage:
    This script is intended for visualizing bike-sharing data provided by the Helsinki region transport (HSL) and
    CityBikeFinland

    The script contains code blocks for the creating several tables and other charts from bike-sharing data

Author:
    Elias Willberg

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from matplotlib import dates
import matplotlib.ticker as plticker
from scipy.cluster.hierarchy import dendrogram, linkage
import datetime as dt

# Set visulization style to Seaborn 
sns.set()
sns.set(font_scale=1.5)

# Read data
BSS_path = 'C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS2017_Full_season_with_DistanceDifference_1_49m.csv'
ManipulatedDF_path = 'C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS_Full_season_1_34m_Loops_RoundTrips_Included.csv'
ManipulatedAllUsersDF_path = 'C:\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS_Full_season_1_49m_Loops_RoundTrips_Included.csv'
CombinedUserGroup_path = 'Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_UsersNewVersion.csv'
AllUsersCombinedUserGroup_path = "Z:\Gradu\Data\CSV\ProcessedData_Full_season\BSS_Full_season_ALLUsersNewVersion.csv"

CombinedUserGroup = pd.read_csv(CombinedUserGroup_path, sep=",", encoding="utf8");
AllUsersCombinedUserGroup = pd.read_csv(AllUsersCombinedUserGroup_path, sep=",", encoding="utf8");
manipulatedDF = pd.read_csv(ManipulatedDF_path, sep=",", encoding="utf8");
manipulatedAllUsersDF = pd.read_csv(ManipulatedAllUsersDF_path, sep=",", encoding="utf8");

"""
manipulatedDF= pd.read_csv(ManipulatedDF_path, sep = ",", encoding  ="utf8");
manipulatedAllUsersDF= pd.read_csv(ManipulatedAllUsersDF_path, sep = ",", encoding  ="utf8");
testSet1 = manipulatedDF.iloc[0:11000] 
"""

# Filter the user group dafaframe to only contain users that have at least X number of trips
FilteredCombinedUserGroup = CombinedUserGroup.loc[CombinedUserGroup['trip_count'] > 9]

# Calculate z-score for numeric columns to remove the outliers in the data
numeric_cols = CombinedUserGroup.select_dtypes(include=[np.number]).columns
numeric_cols = numeric_cols[4:]
CombinedUserGroupWithoutOutliers = CombinedUserGroup[numeric_cols].apply(zscore)
CombinedUserGroupWithoutOutliers = CombinedUserGroupWithoutOutliers[
    ((CombinedUserGroupWithoutOutliers < 4) & (CombinedUserGroupWithoutOutliers > -4)).all(axis=1)]

CombinedUserGroupWithoutOutliers = CombinedUserGroup.loc[CombinedUserGroup["trip_count"] < 600]

# Histogram 1: User Counts by age and gender 
CombinedUserGroup['hsl_gender'].replace('none', np.nan, inplace=True)  # change none values to nan
CombinedUserGroup['age_dec'] = CombinedUserGroup.hsl_age.map(
    lambda hsl_age: 5 * (hsl_age // 5))  # create the age by decade column
CombinedUserGroup['hsl_age'] = CombinedUserGroup.loc[(CombinedUserGroup['hsl_age'] >= 15)]
plt.figure();
h1 = sns.countplot(x="age_dec", hue="hsl_gender", data=CombinedUserGroup, palette=["skyblue", "palevioletred"])

h1.axes.set_title("", fontsize=17, wrap=True, y=1.04)
h1.set_xlabel("Age group", fontsize=14, fontname="Verdana")
h1.set_ylabel("User count", fontsize=14, fontname="Verdana")
h1.legend(fontsize=14)
h1.text(12, 1800, "Males: 54.4 %", size=13, fontname="Verdana")  # male_count = 12 625, female_count = 10 556
h1.text(12, 1600, "Females: 45.6 %", size=13, fontname="Verdana")

h1.set_xticklabels(
    labels=["0", "5", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90"],
    fontname="Verdana")

# Histogram 2: Trip Counts by age and gender
manipulatedDF['hsl_gender'].replace('none', np.nan, inplace=True)  # change none values to nan
manipulatedDF['age_dec'] = manipulatedDF.hsl_age.map(
    lambda hsl_age: 5 * (hsl_age // 5))  # create the age by decade column
plt.figure();
getCountsByGenderAndAge = manipulatedDF.groupby(["age_dec", "hsl_gender"]).count()
h2 = sns.countplot(x="age_dec", hue="hsl_gender", data=manipulatedDF, palette=["skyblue", "palevioletred"])
h2.axes.set_title("", fontsize=17, wrap=True, y=1.04)
h2.set_xlabel("Age group", fontsize=14, fontname="Verdana")
h2.set_ylabel("Trip count", fontsize=14, fontname="Verdana")
h2.legend(fontsize=14)

h2.text(12, 90000, "Males: 59.9 %", size=13, fontname="Verdana")  # male_count = 12 625, female_count = 10 556
h2.text(12, 83000, "Females: 40.1 %", size=13, fontname="Verdana")  # male_count = 547 116, female_count = 366 315

h2.set_xticklabels(
    labels=["0", "5", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90"],
    fontname="Verdana")

# Age variation tables (only those columns included where gender is present to match the table results with histograms 1 and 2)
onlyMenAndWomen = manipulatedDF.loc[(manipulatedDF['hsl_gender'] == "male") | (manipulatedDF['hsl_gender'] == "female")]
ageCounts = onlyMenAndWomen['age_dec'].value_counts()
ageCountsPros = (ageCounts / ageCounts.sum()) * 100
AgeDFTrips = pd.concat([ageCounts, ageCountsPros], axis=1)

onlyMenAndWomen2 = CombinedUserGroupWithoutOutliers.loc[(CombinedUserGroupWithoutOutliers['hsl_gender'] == "male") | (
            CombinedUserGroupWithoutOutliers['hsl_gender'] == "female")]
ageCounts2 = onlyMenAndWomen2['age_dec'].value_counts()
ageCountsPros2 = (ageCounts2 / ageCounts2.sum()) * 100
AgeDFUsers = pd.concat([ageCounts2, ageCountsPros2], axis=1)

# Histogram 4: Trip variation by month, week, and day
manipulatedDF['Week'] = pd.DatetimeIndex(manipulatedDF['departure_time']).week
manipulatedAllUsersDF['week'] = pd.DatetimeIndex(manipulatedAllUsersDF['departure_time']).week

MonthDF = manipulatedDF['id'].groupby(manipulatedDF['Month']).count().reset_index()
WeekDF = manipulatedDF['id'].groupby(manipulatedDF['Week']).count().reset_index()
DailyDF = manipulatedDF.groupby(manipulatedDF['DayOfTheYear']).agg(
    {'id': 'count', 'WeekOrWeekend': 'mean', 'departure_time': 'first'}).reset_index()

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
monthlyHist = fig.add_subplot(grid[0, 0])
weeklyHist = fig.add_subplot(grid[0, 1:])
dailyHist = fig.add_subplot(grid[1, 0:])

b1 = sns.barplot(x='Month', y='id', data=MonthDF, ax=monthlyHist, color='steelblue')
b1.set_xlabel('Month (2017)', fontsize=20)
b1.set_ylabel('Trip count', fontsize=20)

b2 = sns.barplot(x='Week', y='id', data=WeekDF, ax=weeklyHist, color='steelblue')
b2.set_xlabel('Week (2017)', fontsize=20)
b2.set_ylabel('Trip count', fontsize=20)

b3 = sns.barplot(x='departure_time', y='id', data=DailyDF, ax=dailyHist, hue='WeekOrWeekend',
                 palette=["red", "steelblue"])

l = b3.legend()
l.set_title('Weekend / Weekday')

b3.set_xlabel('Day (2017)', fontsize=20)
b3.set_ylabel('Trip count', fontsize=20)
b3_dates = pd.to_datetime(DailyDF['departure_time'][::7]).dt.strftime('%d. %b')

dummyDate = pd.Series({
                          "departure_time": "IGNORE"})  # Due to a bug in MultipleLocator, which does not show tick label at index 0, must add a dummyDate to the index 0
dummyDate = dummyDate.append(b3_dates)
dummyDate = dummyDate.reset_index(drop=True)

b3.set_xticklabels(labels=dummyDate, fontname="Verdana")
for item in b3.get_xticklabels():
    item.set_rotation(90)

b3.xaxis.set_major_locator(plticker.MultipleLocator(7))

fig = plt.figure()
linePlot = fig.add_subplot(111)

UsageByWeekday = manipulatedAllUsersDF['id'].groupby(manipulatedAllUsersDF['weekday']).count().reset_index()
ProportionalUsageByWeekday = (UsageByWeekday / UsageByWeekday.sum()) * 100

weekdayPlot = sns.barplot(x=ProportionalUsageByWeekday.index, y='id', data=ProportionalUsageByWeekday,
                          color='steelblue')
weekdayPlot.set_title('BIKE SHARING TRIPS BY WEEKDAY', fontsize=50)
weekdayPlot.set_ylim(6, 20)
weekdayPlot.set_ylabel('% of all trips ', fontsize=35)
weekdayPlot.set_xlabel('weekday (from Mon)', fontsize=35)
weekdayPlot.set_xticklabels([1, 2, 3, 4, 5, 6, 7], fontsize=25)
weekdayPlot.set_yticklabels([str(int(x)) for x in linePlot.get_yticks([])], fontsize=25)

# Histogram 5: Trips per bike per day by month and week
WeeklyTripCountPerBike = manipulatedAllUsersDF['id'].groupby(manipulatedAllUsersDF['week']).count().reset_index()
WeeklyTripCountPerBike['id'] = WeeklyTripCountPerBike['id'] / 1400  # 1400 is the total number of bikes
ExceptionWeeks = [22, 26, 31, 35, 39]  # the weeks that have only 6 days available in the data
WeeklyTripCountPerBike['id'] = WeeklyTripCountPerBike.apply(
    lambda x: x.id / 2 if x.week == 44 else (x.id / 6 if x.week in ExceptionWeeks else x.id / 7), axis=1)

MonthlyTripCountPerBike = manipulatedAllUsersDF['id'].groupby(manipulatedAllUsersDF['Month']).count().reset_index()
MonthlyTripCountPerBike['id'] = MonthlyTripCountPerBike['id'] / 1400  # 1400 is the total number of bikes
ExceptionMonths = [6, 9]
MonthlyTripCountPerBike['id'] = MonthlyTripCountPerBike.apply(
    lambda x: x.id / 29 if x.Month in ExceptionMonths else x.id / 30, axis=1)

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
monthlyHist = fig.add_subplot(grid[0, 0])
weeklyHist = fig.add_subplot(grid[0, 1:])

b1 = sns.barplot(x='Month', y='id', data=MonthlyTripCountPerBike, ax=monthlyHist, color='steelblue')
b1.set_ylim(0, 10)
b1.set_xlabel('Month (2017)', fontsize=30)
b1.set_ylabel('Trip count day / bike', fontsize=30)
b1.set_yticklabels(b1.get_yticks(), fontsize=23)
b1.set_xticklabels(MonthlyTripCountPerBike.Month, fontsize=23)
b1.set_yticklabels([str(int(x)) for x in b1.get_yticks([])], fontsize=25)

b2 = sns.barplot(x='week', y='id', data=WeeklyTripCountPerBike, ax=weeklyHist, color='steelblue')
b2.set_ylim(0, 10)
b2.set_xlabel('Week (2017)', fontsize=30)
b2.set_ylabel('Trip count day / bike', fontsize=30)
b2.set_xticklabels(WeeklyTripCountPerBike.week, fontsize=23)
b2.set_yticklabels([str(int(x)) for x in b2.get_yticks([])], fontsize=25)

# Lineplot 1: Hourly trip count variation by gender
hourlyTripGroupByGender = manipulatedDF.groupby(["DepHour", "hsl_gender", "WeekOrWeekend"]).count().reset_index()
hourlyTripGroupByGender = hourlyTripGroupByGender.loc[(hourlyTripGroupByGender["hsl_gender"] == "male") | (hourlyTripGroupByGender["hsl_gender"] == "female")]

hourlyTripGroupByWomen = hourlyTripGroupByGender.loc[hourlyTripGroupByGender["hsl_gender"] == "female"]
hourlyTripGroupByMen = hourlyTripGroupByGender.loc[hourlyTripGroupByGender["hsl_gender"] == "male"]

hourlyTripGroupByGender["genderWeekday"] = hourlyTripGroupByGender["hsl_gender"] + "_" + hourlyTripGroupByGender["WeekOrWeekend"].astype(str)
hourlyTripGroupByGender["tripPercent"] = hourlyTripGroupByGender.apply(lambda row:(row["index"] / hourlyTripGroupByWomen["index"].sum())*100 if (row["hsl_gender"] == "female") else (row["index"] / hourlyTripGroupByMen["index"].sum())*100, axis = 1)
ax = sns.lineplot(y = "tripPercent", x = "DepHour", hue = "genderWeekday", data =  hourlyTripGroupByGender, palette =["coral", "coral", "navy" ,"navy"], style = "WeekOrWeekend")
plt.xticks(np.arange(0, 24, step=1))
plt.show()

# Lineplot 2: Hourly trip count variation by home area
hourlyTripGroupByHomeArea = manipulatedDF.groupby(["DepHour", "InsideArea", "WeekOrWeekend"]).count().reset_index()
hourlyTripGroupByHomeArea = hourlyTripGroupByHomeArea.loc[(hourlyTripGroupByHomeArea["InsideArea"] == 0) | (hourlyTripGroupByHomeArea["InsideArea"] == 1)]

hourlyTripGroupByInsideUsers = hourlyTripGroupByHomeArea.loc[hourlyTripGroupByHomeArea["InsideArea"] == 1]
hourlyTripGroupByOutsideUsers = hourlyTripGroupByHomeArea.loc[hourlyTripGroupByHomeArea["InsideArea"] == 0]

hourlyTripGroupByHomeArea["homeAreaWeekday"] = hourlyTripGroupByHomeArea["InsideArea"].astype(str) + "_" + hourlyTripGroupByHomeArea["WeekOrWeekend"].astype(str)
hourlyTripGroupByHomeArea["tripPercent"] = hourlyTripGroupByHomeArea.apply(lambda row:(row["index"] / hourlyTripGroupByInsideUsers["index"].sum())*100
                                            if (row["InsideArea"] == 1)
                                            else (row["index"] / hourlyTripGroupByOutsideUsers["index"].sum())*100, axis = 1)
ax = sns.lineplot(y = "tripPercent", x = "DepHour", hue = "homeAreaWeekday", data =  hourlyTripGroupByHomeArea, palette =["coral", "coral", "navy" ,"navy"], style = "WeekOrWeekend")
plt.xticks(np.arange(0, 24, step=1))
plt.show()

# Lineplot 3: Hourly trip count variation by age group
manipulatedDF['age_dec'] = manipulatedDF.hsl_age.map( lambda hsl_age: 20 * (hsl_age // 20))  # Create age decade column

hourlyTripGroupByAgeGroup = manipulatedDF.groupby(["DepHour", "age_dec", "WeekOrWeekend"]).count().reset_index()
hourlyTripGroupByAgeGroup = hourlyTripGroupByAgeGroup.loc[hourlyTripGroupByAgeGroup["age_dec"] < 80]

hourlyTripGroupByAge_0_20 = hourlyTripGroupByAgeGroup.loc[hourlyTripGroupByAgeGroup["age_dec"] < 20]
hourlyTripGroupByAge_21_40 = hourlyTripGroupByAgeGroup.loc[(hourlyTripGroupByAgeGroup["age_dec"] >= 20) & (hourlyTripGroupByAgeGroup["age_dec"] < 40)]
hourlyTripGroupByAge_41_60 = hourlyTripGroupByAgeGroup.loc[(hourlyTripGroupByAgeGroup["age_dec"] >= 40) & (hourlyTripGroupByAgeGroup["age_dec"] < 60)]
hourlyTripGroupByAge_61_80 = hourlyTripGroupByAgeGroup.loc[(hourlyTripGroupByAgeGroup["age_dec"] >= 60) & (hourlyTripGroupByAgeGroup["age_dec"] < 80)]

hourlyTripGroupByAgeGroup["ageGroupWeekday"] = hourlyTripGroupByAgeGroup["age_dec"].astype(str) + "_" + hourlyTripGroupByAgeGroup["WeekOrWeekend"].astype(str)
hourlyTripGroupByAgeGroup["tripPercent"] = hourlyTripGroupByAgeGroup.apply(lambda row:(row["index"] / hourlyTripGroupByAge_0_20["index"].sum())*100
                                            if (row["age_dec"] == 0)
                                            else ((row["index"] / hourlyTripGroupByAge_21_40["index"].sum())*100
                                                                           if row["age_dec"] == 20
                                                                           else ((row["index"] / hourlyTripGroupByAge_41_60["index"].sum())*100
                                                                                                    if row["age_dec"] == 40
                                                                                                    else ((row["index"] / hourlyTripGroupByAge_61_80["index"].sum())*100))), axis = 1)

ax = sns.lineplot(y = "tripPercent", x = "DepHour", hue = "ageGroupWeekday", data =  hourlyTripGroupByAgeGroup, palette =["coral", "coral", "c" ,"c", "k", "k", "forestgreen", "forestgreen"], style = "WeekOrWeekend")
plt.xticks(np.arange(0, 24, step=1))
plt.show()

# Lineplot 4: Hourly trip count variation by subscription type
hourlyTripGroupByFormula = manipulatedAllUsersDF.groupby(["DepHour", "formula", "WeekOrWeekend"]).count().reset_index()
hourlyTripGroupByFormula = hourlyTripGroupByFormula.loc[(hourlyTripGroupByFormula["formula"] == "Day") | (hourlyTripGroupByFormula["formula"] == "Week") | (hourlyTripGroupByFormula["formula"] == "Year")]

hourlyTripGroupByDayUsers = hourlyTripGroupByFormula.loc[hourlyTripGroupByFormula["formula"] == "Day"]
hourlyTripGroupByWeekUsers = hourlyTripGroupByFormula.loc[hourlyTripGroupByFormula["formula"] == "Week"]
hourlyTripGroupByYearUsers = hourlyTripGroupByFormula.loc[hourlyTripGroupByFormula["formula"] == "Year"]

hourlyTripGroupByFormula["formulaWeekday"] = hourlyTripGroupByFormula["formula"].astype(str) + "_" + hourlyTripGroupByFormula["WeekOrWeekend"].astype(str)
hourlyTripGroupByFormula["tripPercent"] = hourlyTripGroupByFormula.apply(lambda row:(row["index"] / hourlyTripGroupByDayUsers["index"].sum())*100
                                            if (row["formula"] == "Day")
                                            else ((row["index"] / hourlyTripGroupByWeekUsers["index"].sum())*100
                                                                           if row["formula"] == "Week"
                                                                           else ((row["index"] / hourlyTripGroupByYearUsers["index"].sum())*100)), axis = 1)


ax = sns.lineplot(y = "tripPercent", x = "DepHour", hue = "formulaWeekday", data =  hourlyTripGroupByFormula, palette =["coral", "coral", "c" ,"c", "k", "k"], style = "WeekOrWeekend")
plt.xticks(np.arange(0, 24, step=1))
plt.show()

# Lineplot 5: Hourly trip count variation by use activity
AllUsersCombinedUserGroup.sort_values('trip_count', ascending=False, inplace=True)
q = pd.qcut(AllUsersCombinedUserGroup["trip_count"], 4)
AllUsersCombinedUserGroup['ActivityQ'] = q

manipulatedAllUsersDF_merge=manipulatedAllUsersDF.merge(AllUsersCombinedUserGroup,left_on="uid",right_on="uid")

hourlyTripGroupByUseActivity = manipulatedAllUsersDF_merge.groupby(["DepHour","ActivityQ", "WeekOrWeekend"]).count().reset_index()

hourlyTripGroupByActivityQ1 = hourlyTripGroupByUseActivity.loc[hourlyTripGroupByUseActivity["ActivityQ"].astype(str) == "(0.999, 6.0]"]
hourlyTripGroupByActivityQ2 = hourlyTripGroupByUseActivity.loc[hourlyTripGroupByUseActivity["ActivityQ"].astype(str) == "(6.0, 18.0]"]
hourlyTripGroupByActivityQ3 = hourlyTripGroupByUseActivity.loc[hourlyTripGroupByUseActivity["ActivityQ"].astype(str) == "(18.0, 46.0]"]
hourlyTripGroupByActivityQ4 = hourlyTripGroupByUseActivity.loc[hourlyTripGroupByUseActivity["ActivityQ"].astype(str) == "(46.0, 1124.0]"]

hourlyTripGroupByUseActivity["useActivityWeekday"] = hourlyTripGroupByUseActivity["ActivityQ"].astype(str) + "_" + hourlyTripGroupByUseActivity["WeekOrWeekend"].astype(str)

hourlyTripGroupByUseActivity["tripPercent"] = hourlyTripGroupByUseActivity.apply(lambda row:(row["index"] / hourlyTripGroupByActivityQ1["index"].sum())*100
                                            if str(row["ActivityQ"]) == "(0.999, 6.0]"
                                            else ((row["index"] / hourlyTripGroupByActivityQ2["index"].sum())*100
                                                                           if str(row["ActivityQ"]) == "(6.0, 18.0]"
                                                                           else ((row["index"] / hourlyTripGroupByActivityQ3["index"].sum())*100
                                                                                                    if str(row["ActivityQ"]) == "(18.0, 46.0]"
                                                                                                    else ((row["index"] / hourlyTripGroupByActivityQ4["index"].sum())*100))), axis = 1)

ax = sns.lineplot(y = "tripPercent", x = "DepHour", hue = "useActivityWeekday", data =  hourlyTripGroupByUseActivity, palette =["coral", "coral", "c" ,"c", "k", "k", "forestgreen", "forestgreen"], style = "WeekOrWeekend")
plt.xticks(np.arange(0, 24, step=1))

# Histogram 6: Trip time variation
manipulatedDF['duration_group'] = manipulatedDF.duration.map(
    lambda duration: (duration / 60))  # create the age by decade column
manipulatedDF['duration_group'] = manipulatedDF.duration_group.map(
    lambda duration: 1 * (duration // 1))  # create the age by decade column
sns.set(font_scale=1.7)

fig = plt.figure(figsize=(6, 6))
TimeDF = manipulatedDF['id'].groupby(manipulatedDF['duration_group']).count().reset_index()
TimeDFUnder60 = TimeDF.loc[:59]
TimeDFUnder60['duration_group'] = TimeDFUnder60['duration_group'].astype(int)
b4 = sns.barplot(x='duration_group', y='id', data=TimeDFUnder60, color='steelblue')
b4.set_xlabel('Duration (min)', fontsize=30)
b4.set_ylabel('Trip count', fontsize=30)

b4.set_yticklabels([str(int(x)) for x in b4.get_yticks([])], fontsize=25)

b4.text(12, 90000, "Trip duration 0 - 30 min: 96.7 %", size=25,
        fontname="Verdana")  # male_count = 12 625, female_count = 10 556
b4.text(12, 83000, "Trip duration over 30 min: 3,3 %", size=25,
        fontname="Verdana")  # male_count = 547 116, female_count = 366 315

# Chart 1 showing users season type variation
SeasonTypeByTrip = manipulatedAllUsersDF['formula'].value_counts()
SeasonTypeByUser = manipulatedAllUsersDF['uid'].groupby(manipulatedAllUsersDF['formula']).nunique()
SeasonTypeByTrip.plot()

# Histogram 8 Showing the histogram for trip count, tripsPerDay and the number of unique trip days columns
CombinedUserGroup['trip_count'].describe()
CombinedUserGroup['tripsPerDay'].describe()
CombinedUserGroup['DayOfTheYear_nunique'].describe()

fig = plt.figure(figsize=(12, 3))
grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.2)

tripCountHist = fig.add_subplot(grid[0, 0])
tripCountHist.hist(CombinedUserGroup['trip_count'], range=(0, 400), bins=20)
tripCountHist.set_xlabel('Trip count', fontsize=20)
tripCountHist.set_ylabel('User count', fontsize=20)
tripCountHist.text(250, 4000, "mean: 38.3 \nstd: 46.3 \nmin: 1 \n25%: 8 \n50%: 22 \n75% 50 \nmax: 1124", size=15,
                   fontname="Verdana", weight="bold")

tripsPerDayHist = fig.add_subplot(grid[0, 1])
tripsPerDayHist.hist(CombinedUserGroup['DayOfTheYear_nunique'], range=(0, 160), bins=20, color="darkslategray")
tripsPerDayHist.set_xlabel('Unique user days', fontsize=20)
tripsPerDayHist.set_ylabel('User count', fontsize=20)
tripsPerDayHist.text(100, 3000, "mean: 22.6 \nstd: 23.0 \nmin: 1 \n25%: 6 \n50%: 15 \n75% 32 \nmax: 158", size=15,
                     fontname="Verdana", weight="bold")

tripsPerDayHist = fig.add_subplot(grid[0, 2])
tripsPerDayHist.hist(CombinedUserGroup['tripsPerDay'], range=(0, 3), bins=20, color="indianred")
tripsPerDayHist.set_xlabel('Trips per day ', fontsize=20)
tripsPerDayHist.set_ylabel('User count', fontsize=20)
tripsPerDayHist.text(2.0, 5000, "mean: 0.22 \nstd: 0.26 \nmin: 0.005 \n25%: 0.05 \n50%: 0.13 \n75% 0.29 \nmax: 6.42",
                     size=15, fontname="Verdana", weight="bold")

CombinedUserGroup['PT_trip_pros'].hist(bins=20)
CombinedUserGroup['userDayRatio'].hist(bins=20)
CombinedUserGroup['userDayCount'].hist(bins=20)

# Cumulative trip count plot: 
CombinedUserGroup.sort_values('trip_count', ascending=False, inplace=True)
CombinedUserGroup['cum_sum'] = CombinedUserGroup.trip_count.cumsum()
CombinedUserGroup['cum_perc'] = 100 * CombinedUserGroup.cum_sum / CombinedUserGroup.trip_count.sum()

sns.set(font_scale=2.1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Cumulative percentage of users % ', fontsize=35)
ax.set_ylabel('Cumulative percentage of trips %', fontsize=35)

CombinedUserGroup = CombinedUserGroup.sort_values('cum_sum', ascending=True)
CombinedUserGroup['seq'] = range(1, len(CombinedUserGroup) + 1)
CombinedUserGroup['user_perc'] = 100 * CombinedUserGroup.seq / len(CombinedUserGroup.seq)

ax.plot(CombinedUserGroup['user_perc'], CombinedUserGroup['cum_perc'])
percentage = 0.2
while percentage < 1.1:
    point = int(round(len(CombinedUserGroup) * percentage, 0))
    ax.text(CombinedUserGroup.iloc[point]["user_perc"], CombinedUserGroup.iloc[point]["cum_perc"],
            "User count: %s \n Trip count: %s" % (
            CombinedUserGroup.iloc[point]["seq"], CombinedUserGroup.iloc[point]["cum_sum"]),
            horizontalalignment='left',
            verticalalignment='top',
            multialignment='center')
    ax1 = fig.add_subplot(111)
    ax1.plot([CombinedUserGroup.iloc[point]["user_perc"]], [CombinedUserGroup.iloc[point]["cum_perc"]], marker='o',
             markersize=10, markerfacecolor="darkred")
    percentage += 0.2

plt.show()

# In/Out Visualizations:  Inside BSS coverage area users vs Outside BSS coverage area users 
sns.set(font_scale=1)

insiders = CombinedUserGroup.loc[CombinedUserGroup['insideArea'] == 1]
outsiders = CombinedUserGroup.loc[CombinedUserGroup['insideArea'] == 0]

FilteredCombinedUserGroup = CombinedUserGroup.loc[
    (CombinedUserGroup['trip_count'] > 0) & (CombinedUserGroup['diff_median'])]

fig = plt.figure()
boxPlot = fig.add_subplot(111)

boxPlot = sns.boxplot(x="insideArea", y="diff_median", data=FilteredCombinedUserGroup, width=0.5)
boxPlot.set_ylim(-400, 30000)
boxPlot.set_xlabel('')
boxPlot.set_title('MEDIAN ROUTE DISTANCE DIFFERENCE \n', fontsize=50)
boxPlot.text(-0.1, 1020, "(shortest route distance - realized route distance)", fontsize=35)
boxPlot.set_ylabel('(m) ', fontsize=35)
boxPlot.set_xticklabels(
    ['0 \n \n Users living inside BSS coverage area', '1 \n \n Users living outside BSS coverage area'], fontsize=30)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=30)
plt.show()

fig = plt.figure()
boxPlot = fig.add_subplot(111)

boxPlot = sns.boxplot(x="insideArea", y="PT_trip_pros", data=CombinedUserGroup, width=0.5)
boxPlot.set_title('POTENTIAL PUBLIC TRANSPORT CHAIN TRIPS \n ', fontsize=50)
boxPlot.text(-0.25, 1.015, "(Departure or return station in the immediate vicinity of a metro or train station)",
             fontsize=30)
boxPlot.set_ylabel('% of all trips ', fontsize=35)
boxPlot.set_xlabel('')
boxPlot.set_xticklabels(
    ['0 \n \n Users living inside BSS coverage area', '1 \n \n Users living outside BSS coverage area'], fontsize=30)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=30)
plt.show()

fig = plt.figure()
boxPlot = fig.add_subplot(111)

boxPlot = sns.boxplot(x="insideArea", y="hsl_age", data=CombinedUserGroup, width=0.5)
boxPlot.set_title('USER AGE VARIATION ', fontsize=50)
boxPlot.set_ylabel('User age (years) ', fontsize=35)
boxPlot.set_xlabel('')
boxPlot.set_xticklabels(
    ['0 \n \n Users living inside BSS coverage area', '1 \n \n Users living outside BSS coverage area'], fontsize=30)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=30)
plt.show()

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot = sns.boxplot(x="insideArea", y="week_weekend_relaRatio", data=CombinedUserGroup, width=0.5)
boxPlot.set_title('WEEKDAY VS WEEKEND RELATIVE USE RATIO  \n', fontsize=50)
boxPlot.text(-0.2, 20.2, "(The number of weekday trips per one weekend trip)", fontsize=35)
boxPlot.set_ylim(-0.2, 20)
boxPlot.set_ylabel('ratio ', fontsize=33)
boxPlot.set_xlabel('')
boxPlot.set_xticklabels(
    ['0 \n \n Users living inside BSS coverage area', '1 \n \n Users living outside BSS coverage area'], fontsize=28)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=30)

plt.show()

boxPlot = sns.boxplot(x="insideArea", y="week_minus_weekend", data=CombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="insideArea", y="Days_RetToStartDep_ratio", data=CombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="insideArea", y="depStartFromRet_ratio", data=CombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="insideArea", y="DepStatSTD", data=CombinedUserGroup, width=0.5)
boxPlot.set_ylim(-0, 20)

boxPlot = sns.boxplot(x="insideArea", y="speed_median", data=CombinedUserGroup, width=0.5)  # speed difference

# Age Visualizations:  Box plots for different age groups  
FilteredCombinedUserGroup = CombinedUserGroup.loc[
    (CombinedUserGroup['hsl_age'] > 15) & (CombinedUserGroup['hsl_age'] < 80)]
FilteredCombinedUserGroup['age_dec'] = CombinedUserGroup.hsl_age.map(
    lambda hsl_age: 10 * (hsl_age // 10))  # create the age by decade column

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 2000)
boxPlot = sns.boxplot(x="age_dec", y="week_weekend_relaRatio", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="age_dec", y="Days_RetToStartDep_ratio", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="age_dec", y="depStartFromRet_ratio", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="age_dec", y="DepStatSTD", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="age_dec", y="PT_trip_pros", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="age_dec", y="diff_median", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="age_dec", y="trip_count", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="age_dec", y="speed_median", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="age_dec", y="duration_median", data=FilteredCombinedUserGroup, width=0.5)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 200)
boxPlot = sns.boxplot(x="age_dec", y="trip_count", data=FilteredCombinedUserGroup, width=0.5)
boxPlot.set_title('TRIP COUNT', fontsize=50)
boxPlot.set_ylabel('number of trips ', fontsize=35)
boxPlot.set_xlabel('AGE GROUP', fontsize=35)
boxPlot.set_yticklabels([str(int(x)) for x in boxPlot.get_yticks([])], fontsize=25)
boxPlot.set_xticklabels(["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"], fontsize=25)

FilteredCombinedUserGroup = FilteredCombinedUserGroup.loc[(FilteredCombinedUserGroup['trip_count'] > 4)]
fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-400, 1000)
boxPlot = sns.boxplot(x="age_dec", y="diff_median", data=FilteredCombinedUserGroup, width=0.5)
boxPlot.set_title('MEDIAN ROUTE DISTANCE DIFFERENCE \n', fontsize=50)
boxPlot.text(0.7, 1020, "(shortest route distance - realized route distance)", fontsize=35)
boxPlot.set_ylabel('(m) ', fontsize=35)
boxPlot.set_xlabel('AGE GROUP', fontsize=35)
boxPlot.set_yticklabels([str(int(x)) for x in boxPlot.get_yticks([])], fontsize=25)
boxPlot.set_xticklabels(["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"], fontsize=25)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 2000)
boxPlot = sns.boxplot(x="age_dec", y="duration_median", data=FilteredCombinedUserGroup, width=0.5)
boxPlot.set_title('MEDIAN TRIP DURATION BY AGE GROUPS ', fontsize=50)
boxPlot.set_ylabel('Median trip duration (s)', fontsize=35)
boxPlot.set_xlabel('AGE GROUP', fontsize=35)
boxPlot.set_yticklabels([str(int(x)) for x in boxPlot.get_yticks([])], fontsize=25)
boxPlot.set_xticklabels(["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"], fontsize=25)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(4, 16)
boxPlot = sns.boxplot(x="age_dec", y="speed_median", data=FilteredCombinedUserGroup, width=0.5)
boxPlot.set_title('MEDIAN TRIP SPEED BY AGE GROUPS ', fontsize=50)
boxPlot.set_ylabel('Median trip speed (km/h)', fontsize=35)
boxPlot.set_xlabel('AGE GROUP', fontsize=35)
boxPlot.set_yticklabels([str(int(x)) for x in boxPlot.get_yticks([])], fontsize=25)
boxPlot.set_xticklabels(["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"], fontsize=25)

# Gender Visualizations:  Box plots for different Gender groups  
FilteredCombinedUserGroup = CombinedUserGroup.loc[(CombinedUserGroup['hsl_gender'] != "none")]

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 20)

boxPlot = sns.boxplot(x="hsl_gender", y="week_weekend_relaRatio", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="hsl_gender", y="Days_RetToStartDep_ratio", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="hsl_gender", y="depStartFromRet_ratio", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="hsl_gender", y="DepStatSTD", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="hsl_gender", y="PT_trip_pros", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="hsl_gender", y="diff_median", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="hsl_gender", y="trip_count", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="hsl_gender", y="speed_median", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="hsl_gender", y="duration_median", data=FilteredCombinedUserGroup, width=0.5)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 200)
boxPlot = sns.boxplot(x="hsl_gender", y="trip_count", data=FilteredCombinedUserGroup, width=0.5,
                      palette=("peru", "indianred"))
boxPlot.set_title('TRIP COUNT BY GENDER', fontsize=50)
boxPlot.set_ylabel('number of trips ', fontsize=35)
boxPlot.set_xlabel('Gender', fontsize=35)
boxPlot.set_yticklabels([str(int(x)) for x in boxPlot.get_yticks([])], fontsize=25)
boxPlot.set_xticklabels(['Male', 'Female'], fontsize=28)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 5000)
boxPlot = sns.boxplot(x="hsl_gender", y="distance_median", data=FilteredCombinedUserGroup, width=0.5,
                      palette=("peru", "indianred"))
boxPlot.set_title('MEDIAN TRIP DISTANCE BY GENDER ', fontsize=50)
boxPlot.set_ylabel('Median trip distance (m)', fontsize=35)
boxPlot.set_xlabel('Gender', fontsize=35)
boxPlot.set_yticklabels([str(int(x)) for x in boxPlot.get_yticks([])], fontsize=25)
boxPlot.set_xticklabels(['Male', 'Female'], fontsize=28)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 2000)
boxPlot = sns.boxplot(x="hsl_gender", y="duration_median", data=FilteredCombinedUserGroup, width=0.5,
                      palette=("peru", "indianred"))
boxPlot.set_title('MEDIAN TRIP DURATION BY GENDER ', fontsize=50)
boxPlot.set_ylabel('Median trip duration (s)', fontsize=35)
boxPlot.set_xlabel('Gender', fontsize=35)
boxPlot.set_yticklabels([str(int(x)) for x in boxPlot.get_yticks([])], fontsize=25)
boxPlot.set_xticklabels(['Male', 'Female'], fontsize=28)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 1)
boxPlot = sns.boxplot(x="hsl_gender", y="PT_trip_pros", data=FilteredCombinedUserGroup, width=0.5,
                      palette=("peru", "indianred"))
boxPlot.set_title('POTENTIAL PUBLIC TRANSPORT CHAIN TRIPS\n ', fontsize=50)
boxPlot.text(-0.25, 1.015, "(Departure or return station in the immediate vicinity of a metro or train station)",
             fontsize=30)
boxPlot.set_ylabel('% of all trips ', fontsize=35)
boxPlot.set_xlabel('Gender', fontsize=35)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=30)
boxPlot.set_xticklabels(['Male', 'Female'], fontsize=28)

boxPlot.set_ylim(-400, 1000)

# Activity Visualizations:  Box plots for regular and sporaric users  
FilteredCombinedUserGroup.sort_values('trip_count', ascending=False, inplace=True)
q = pd.qcut(FilteredCombinedUserGroup["trip_count"], 5)
FilteredCombinedUserGroup['ActivityQ'] = q

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 2000)
boxPlot = sns.boxplot(x="ActivityQ", y="week_weekend_relaRatio", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="ActivityQ", y="Days_RetToStartDep_ratio", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="ActivityQ", y="depStartFromRet_ratio", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="ActivityQ", y="DepStatSTD", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="ActivityQ", y="PT_trip_pros", data=FilteredCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="ActivityQ", y="duration_median", data=FilteredCombinedUserGroup, width=0.5)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, )
boxPlot = sns.boxplot(x="ActivityQ", y="depStartFromRet_ratio", data=FilteredCombinedUserGroup, width=0.5)
boxPlot.set_title('SHARE OF CHAINED TRIPS BY TRIP COUNT QUANTILES \n', fontsize=50)
boxPlot.text(-0, 1, "(Where the departure station has been the return station of the previous trip", fontsize=30)
boxPlot.set_ylabel('Proportion of chained trips (%)', fontsize=35)
boxPlot.set_xlabel('User quantiles by trip count ', fontsize=35)
boxPlot.text(0.1, -0.075,
             "<-- light users                                                                            heavy users -->",
             fontsize=30)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=25)
boxPlot.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"], fontsize=25)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 30)
boxPlot = sns.boxplot(x="ActivityQ", y="DepStatSTD", data=FilteredCombinedUserGroup, width=0.5)
boxPlot.set_title('STANDARD DEVIATION OF USER\'S STATION USAGE', fontsize=50)
boxPlot.set_ylabel('Standard deviation', fontsize=35)
boxPlot.set_xlabel('User quantiles by trip count ', fontsize=35)
boxPlot.text(0.1, -2.2,
             "<-- light users                                                                            heavy users -->",
             fontsize=30)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=25)
boxPlot.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"], fontsize=25)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 30)
boxPlot = sns.boxplot(x="ActivityQ", y="DepStatSTD", data=FilteredCombinedUserGroup, width=0.5)
boxPlot.set_title('STANDARD DEVIATION OF USER\'S STATION USAGE', fontsize=50)
boxPlot.set_ylabel('Standard deviation', fontsize=35)
boxPlot.set_xlabel('User quantiles by trip count ', fontsize=35)
boxPlot.text(0.1, -2.2,
             "<-- light users                                                                            heavy users -->",
             fontsize=30)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=25)
boxPlot.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"], fontsize=25)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 2000)
boxPlot = sns.boxplot(x="ActivityQ", y="duration_median", data=FilteredCombinedUserGroup, width=0.5)
boxPlot.set_title('MEDIAN TRIP DURATION BY TRIP COUNT QUANTILES ', fontsize=50)
boxPlot.set_ylabel('Median trip duration (s)', fontsize=35)
boxPlot.set_xlabel('User quantiles by trip count ', fontsize=35)
boxPlot.text(0.1, -150,
             "<-- light users                                                                            heavy users -->",
             fontsize=30)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=25)
boxPlot.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"], fontsize=25)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 20)
boxPlot = sns.boxplot(x="ActivityQ", y="speed_median", data=FilteredCombinedUserGroup, width=0.5)
boxPlot.set_title('MEDIAN TRIP SPEED BY TRIP COUNT QUANTILES ', fontsize=50)
boxPlot.set_ylabel('Median trip speed (km/h)', fontsize=35)
boxPlot.set_xlabel('User quantiles by trip count ', fontsize=35)
boxPlot.text(0.1, -1.5,
             "<-- light users                                                                            heavy users -->",
             fontsize=30)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=25)
boxPlot.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"], fontsize=25)

# Formula Visualizations:  Box plots for year, week and day users  
AllUsersCombinedUserGroup.sort_values('trip_count', ascending=False, inplace=True)
AllUsersCombinedUserGroup['tripsPerUserDay'] = AllUsersCombinedUserGroup.apply(
    lambda row: row['trip_count'] if (row['formula'] == 'Day') else (
        row['trip_count'] / 7 if (row['formula'] == 'Week') else (row['trip_count'] / 177)), axis=1)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-500, 2000)
boxPlot = sns.boxplot(x="formula", y="diff_median", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="formula", y="DayOfTheYear_nunique", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="formula", y="DepStatSTD", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="formula", y="PT_trip_pros", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="formula", y="tripsPerUserDay", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="formula", y="depStartFromRet_ratio", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot = sns.boxplot(x="formula", y="trip_count", data=AllUsersCombinedUserGroup, width=0.5)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 25)
boxPlot = sns.boxplot(x="formula", y="tripsPerUserDay", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot.set_title('MEDIAN NUMBER OF TRIPS PER DAY BY THE SUBSCRIPTION TYPE\n', fontsize=40)
boxPlot.text(0, 25, "(Trip count / lenght of the user's subscription in days)", fontsize=35)
boxPlot.set_ylabel('trips per day ', fontsize=35)
boxPlot.set_xlabel('Subscription type', fontsize=35)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=25)
boxPlot.set_xticklabels(["Year", "Week", "Day"], fontsize=25)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 30)
boxPlot = sns.boxplot(x="formula", y="speed_median", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot.set_title('MEDIAN TRIP SPEED BY THE SUBSCRIPTION TYPE  ', fontsize=47)
boxPlot.set_ylabel('speed (km/h) ', fontsize=35)
boxPlot.set_xlabel('Subscription type', fontsize=35)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=25)
boxPlot.set_xticklabels(["Year", "Week", "Day"], fontsize=25)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 5000)
boxPlot = sns.boxplot(x="formula", y="duration_median", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot.set_title('MEDIAN TRIP DURATION BY THE SUBSCRIPTION TYPE  ', fontsize=47)
boxPlot.set_ylabel('duration (s) ', fontsize=35)
boxPlot.set_xlabel('Subscription type', fontsize=35)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=25)
boxPlot.set_xticklabels(["Year", "Week", "Day"], fontsize=25)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-0, 15000)
boxPlot = sns.boxplot(x="formula", y="distance_median", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot.set_title('MEDIAN TRIP DISTANCE BY THE SUBSCRIPTION TYPE  ', fontsize=47)
boxPlot.set_ylabel('distance (m) ', fontsize=35)
boxPlot.set_xlabel('Subscription type', fontsize=35)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=25)
boxPlot.set_xticklabels(["Year", "Week", "Day"], fontsize=25)

fig = plt.figure()
boxPlot = fig.add_subplot(111)
boxPlot.set_ylim(-500, 2000)
boxPlot = sns.boxplot(x="formula", y="diff_median", data=AllUsersCombinedUserGroup, width=0.5)
boxPlot.set_title('MEDIAN ROUTE DISTANCE DIFFERENCE \n', fontsize=50)
boxPlot.text(0, 2020, "(shortest route distance - realized route distance)", fontsize=35)
boxPlot.set_ylabel('distance difference (m) ', fontsize=35)
boxPlot.set_xlabel('Subscription type', fontsize=35)
boxPlot.set_yticklabels(boxPlot.get_yticks(), fontsize=25)
boxPlot.set_xticklabels(["Year", "Week", "Day"], fontsize=25)

plt.show()

# Boxplot 1. Boxplot showing variation of insiders/outsiders 
CombinedUserGroup['age_dec'] = CombinedUserGroup.hsl_age.map(
    lambda hsl_age: 10 * (hsl_age // 10))  # create the age by decade column
CombinedUserGroup['hsl_city_Categ'] = CombinedUserGroup['hsl_city'].apply(
    lambda row: 1 if (row == "Helsinki") else (2 if (row == "Espoo") else (3 if (row == "Vantaa") else 4)))

CombinedUserGroup["week_minus_weekend"] = (CombinedUserGroup["weekdayTripCount"] / 5) - (
            CombinedUserGroup["weekendTripCount"] / 2)
CombinedUserGroup["week_minus_weekendABS"] = (CombinedUserGroup["weekdayTripCount"]) - (
CombinedUserGroup["weekendTripCount"])
CombinedUserGroup["week_weekend_relaRatio"] = CombinedUserGroup.apply(
    lambda row: row['weekdayTripCount'] / 5 if (row['weekendTripCount'] == 0)
    else (row['weekendTripCount'] / 2 if (row['weekdayTripCount'] == 0) else (row['weekdayTripCount'] / 5) / (
                row['weekendTripCount'] / 2)), axis=1)

fig = plt.figure()
boxPlot2 = fig.add_subplot(111)
boxPlot = sns.boxplot(x="insideArea", y="week_weekend_relaRatio", data=CombinedUserGroup, width=0.5)

# MAKE BOXPLOTS OUT OF THESE
CombinedUserGroup['age_dec'] = CombinedUserGroup.loc[
    CombinedUserGroup['age_dec'] > 10]  # create the age by decade column
fig = plt.figure()
boxPlot2 = fig.add_subplot(111)

boxPlot2 = sns.boxplot(x="hsl_city_Categ", y="PT_trip_pros", data=CombinedUserGroup)
