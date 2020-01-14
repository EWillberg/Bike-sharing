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
from matplotlib.legend import Legend


# Set visulization style to Seaborn
sns.set()
sns.set(font_scale=1.5)

# Read data
ManipulatedDF_path = 'C:\LocalData\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS_Full_season_1_34m_Loops_RoundTrips_Included.csv'
ManipulatedAllUsersDF_path = 'C:\LocalData\HY-Data\ELWI\BikeSharingData\Processed2017\CSV\ProcessedData_Full_season\BSS_Full_season_1_49m_Loops_RoundTrips_Included.csv'
CombinedUserGroup_path = 'Z:\\2019\\2019_Gradu\\Data\\CSV\\ProcessedData_Full_season\\BSS_Full_season_UsersNewVersion.csv'
AllUsersCombinedUserGroup_path = 'Z:\\2019\\2019_Gradu\\Data\\CSV\\ProcessedData_Full_season\\BSS_Full_season_ALLUsersNewVersion.csv'
PopulationByAgeGroupHelsinki = "Z:\Gradu\Data\Excel\VaestoHelsinki.csv"

CombinedUserGroup = pd.read_csv(CombinedUserGroup_path, sep=",", encoding="utf8");
AllUsersCombinedUserGroup = pd.read_csv(AllUsersCombinedUserGroup_path, sep=",", encoding="utf8");
manipulatedDF = pd.read_csv(ManipulatedDF_path, sep=",", encoding="utf8");
manipulatedAllUsersDF = pd.read_csv(ManipulatedAllUsersDF_path, sep=",", encoding="utf8");
popHelsinki = pd.read_csv(PopulationByAgeGroupHelsinki, sep=";", encoding="utf8");

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
Over15_DF = manipulatedDF.loc[manipulatedDF["age_dec"] >= 15]

plt.figure()
h2 = sns.countplot(x="age_dec", hue="hsl_gender", data=Over15_DF, palette=["skyblue", "palevioletred"])
h2.axes.set_title("", fontsize=17, wrap=True, y=1.04)
h2.set_xlabel("Age group", fontsize=14, fontname="Verdana")
h2.set_ylabel("Trip count", fontsize=14, fontname="Verdana")
h2.legend(fontsize=14)

h2.text(12, 90000, "Males: 59.9 %", size=13, fontname="Verdana")  # male_count = 12 625, female_count = 10 556
h2.text(12, 83000, "Females: 40.1 %", size=13, fontname="Verdana")  # male_count = 547 116, female_count = 366 315

h2.set_xticklabels(
    labels=["15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90"],
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

# Histogram 3: User count by age and gender with Helsinki demographics included
CombinedUserGroup['age_dec'] = CombinedUserGroup.hsl_age.map(
    lambda hsl_age: 5 * (hsl_age // 5))  # create the age by decade column

over15_Users_DF = CombinedUserGroup.loc[CombinedUserGroup["age_dec"] >= 15]
getUserCountsByGenderAndAge = over15_Users_DF.groupby(["age_dec", "hsl_gender"]).count().reset_index()
getUserCountsByGenderAndAge["normUserCount"] = getUserCountsByGenderAndAge["uid"] / getUserCountsByGenderAndAge[
    "uid"].sum() * 100

popHelsinki["normPopCount"] = popHelsinki["Pop"] / popHelsinki["Pop"].sum() * 100

ageGroup = popHelsinki["Age"].where(popHelsinki["Gender"] == "female").dropna()
womenPop = popHelsinki["normPopCount"].where(popHelsinki["Gender"] == "female").dropna()
menPop = popHelsinki["normPopCount"].where(popHelsinki["Gender"] == "male").dropna()
womenUserCount = getUserCountsByGenderAndAge["normUserCount"].where(
    getUserCountsByGenderAndAge["hsl_gender"] == "female").dropna()
womenUserCount = womenUserCount.append(pd.Series([0, 0, 0]), ignore_index=True)
menTripCount = getUserCountsByGenderAndAge["normUserCount"].where(
    getUserCountsByGenderAndAge["hsl_gender"] == "male").dropna()
menTripCount = menTripCount.append(pd.Series([0]), ignore_index=True)
combinedUserDF = pd.DataFrame(list(zip(ageGroup, womenPop, menPop, womenUserCount, menTripCount)),
                              columns=["age", "womenPopPros", "menPopPros", "womenUserCountPros", "menUserCountPros"])
sns.set_palette("deep")

fig = plt.figure()
ax = combinedUserDF[["menUserCountPros", 'womenUserCountPros']].plot(kind='bar', use_index=True, alpha=0.8, width=0.85)
ax.legend(['Bike-sharing users: Male', 'Bik-sharing users: Female'], bbox_to_anchor=(1, 1), fontsize=30, frameon=False)
ax2 = ax.twiny()
ax2.plot(combinedUserDF[["menPopPros", "womenPopPros"]].values, linestyle='--', marker='o', linewidth=2.5)
ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False, bottom=False,
                labelbottom=False)
ax2.legend(['Total Helsinki population: Male', 'Total Helsinki population: Female'], bbox_to_anchor=(1, 0.75),
           fontsize=30, frameon=False)
ax2.grid(False)

l1 = ax2.lines[0]  # Get the lines to shade the area under line
l2 = ax2.lines[1]

x1 = l1.get_xydata()[:, 0]  # Get the xy data from the lines so that we can shade
y1 = l1.get_xydata()[:, 1]
x2 = l2.get_xydata()[:, 0]
y2 = l2.get_xydata()[:, 1]
ax.fill_between(x1, y1, color="gray", alpha=0.35)
ax.fill_between(x2, y2, color="gray", alpha=0.35)
ax.set_ylabel("Share of total (%)", fontsize=30)
ax.tick_params(axis='y', which='major', labelsize=30)
ax.set_xticklabels(
    labels=["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
            "75-79", "80-84", "85-89", ">90"],
    fontname="Verdana", fontsize="24")

# Histogram 4: Trip count by age and gender with Helsinki demographics included
getTripCountsByGenderAndAge = Over15_DF.groupby(["age_dec", "hsl_gender"]).count().reset_index()
getTripCountsByGenderAndAge["normTripCount"] = getTripCountsByGenderAndAge["index"] / getTripCountsByGenderAndAge[
    "index"].sum() * 100
popHelsinki["normPopCount"] = popHelsinki["Pop"] / popHelsinki["Pop"].sum() * 100

ageGroup = popHelsinki["Age"].where(popHelsinki["Gender"] == "female").dropna()
womenPop = popHelsinki["normPopCount"].where(popHelsinki["Gender"] == "female").dropna()
menPop = popHelsinki["normPopCount"].where(popHelsinki["Gender"] == "male").dropna()
womenTripCount = getTripCountsByGenderAndAge["normTripCount"].where(
    getTripCountsByGenderAndAge["hsl_gender"] == "female").dropna()
womenTripCount = womenTripCount.append(pd.Series([0, 0, 0]), ignore_index=True)
menTripCount = getTripCountsByGenderAndAge["normTripCount"].where(
    getTripCountsByGenderAndAge["hsl_gender"] == "male").dropna()
menTripCount = menTripCount.append(pd.Series([0]), ignore_index=True)
combinedTripDF = pd.DataFrame(list(zip(ageGroup, womenPop, menPop, womenTripCount, menTripCount)),
                              columns=["age", "womenPopPros", "menPopPros", "womenTripCountPros", "menTripCountPros"])

fig = plt.figure()
ax = combinedTripDF[["menTripCountPros", 'womenTripCountPros']].plot(kind='bar', use_index=True, alpha=0.8, width=0.85)
ax.legend(['Bike-sharing trips: Male', 'Bike-sharing trips: Female'], bbox_to_anchor=(1, 1), fontsize=30, frameon=False)
ax2 = ax.twiny()
ax2.plot(combinedTripDF[["menPopPros", "womenPopPros"]].values, linestyle='--', marker='o', linewidth=2.5)
ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False, bottom=False,
                labelbottom=False)
ax2.legend(['Total Helsinki population: Male', 'Total Helsinki population: Female'], bbox_to_anchor=(1, 0.75),
           fontsize=30, frameon=False)
ax2.grid(False)

# Get the lines to shade the area under line
l1 = ax2.lines[0]
l2 = ax2.lines[1]

# Get the xy data from the lines so that we can shade
x1 = l1.get_xydata()[:, 0]
y1 = l1.get_xydata()[:, 1]
x2 = l2.get_xydata()[:, 0]
y2 = l2.get_xydata()[:, 1]
ax.fill_between(x1, y1, color="gray", alpha=0.35)
ax.fill_between(x2, y2, color="gray", alpha=0.35)
ax.set_ylabel("Share of total (%)", fontsize=30)
ax.tick_params(axis='y', which='major', labelsize=30)
ax.set_xticklabels(
    labels=["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
            "75-79", "80-84", "85-89", ">90"],
    fontname="Verdana", fontsize="25")

# Donut plot 1 : Home area chart
"""
In[40] manipulatedDF["InsideArea"].value_counts()
Out[40]: 
1    1067016 -> 79.17%
0     280729 -> 20.83%

In [86] CombinedUserGroup["insideArea"].value_counts()
Out[86]: 
1    24019 -> 69.37%
0    10603 -> 30.64%

Total population in Helsinki inside/outside the BSS station area 
1 269047 -> 39.98
0 403920 +> 60.02
"""
size_of_HomeAreaGroups = [79.17, 20.83]
size_of_HomeAreaGroups2 = [69.37, 30.64]
size_of_HomeAreaGroups3 = [39.98, 60.02]

cmap = plt.get_cmap("tab20b")
colors = cmap(np.array([0,13]))
labels = ["Population in Helsinki living \ninside BSS coverage area",
          "Population in Helsinki living \noutside BSS coverage area"]

fig, ax = plt.subplots()
ax.pie(size_of_HomeAreaGroups, radius=1, colors=colors,
       wedgeprops=dict(width=0.2, edgecolor='w'), autopct='%1.0f%%', pctdistance=1.13,
       textprops={'fontsize': 30, "fontweight": "bold"})
ax.pie(size_of_HomeAreaGroups2, radius=0.8, colors=colors,
       wedgeprops=dict(width=0.2, edgecolor='w'), autopct='%1.0f%%', pctdistance=0.88,
       textprops={'fontsize': 30, "color": "white", "fontweight": "bold"})
ax.pie(size_of_HomeAreaGroups3, radius=0.6, colors=colors,
       wedgeprops=dict(width=0.2, edgecolor='w'), autopct='%1.0f%%', pctdistance=0.45,
       textprops={'fontsize': 30, "fontweight": "bold"})
plt.legend(labels, title="Outermost ring = BSS trips  \nMiddle ring = BSS users "
                         "\nInnermost ring = Helsinki population \n\n      Home area",
           bbox_to_anchor=(0.9, 0.9), fontsize=24)
ax.get_legend().get_title().set_fontsize('24')
ax.set(aspect="equal")

plt.show()

# Donut plot 2 : Subscription type
"""
In [230]: manipulatedAllUsersDF["formula"].value_counts()
Out[230]: 
Year    1375583 -> 91.90%
Day       78711 -> 5.26%
Week      42522 -> 2.84%

In [231]:AllUsersCombinedUserGroup["formula"].value_counts()
Out[231]: 
Year    33557 -> 82.43%
Week     1614 -> 3.97%
Day      5538 -> 13.60%

"""

size_of_SubscriptionGroups1 = [91.90, 5.26,2.84 ]
size_of_SubscriptionGroups2 = [82.43, 3.97, 13.60]

cmap = plt.get_cmap("tab20b")
colors = cmap(np.array([12, 1,10 ]))
labels = ["Year", "Week", "Day"]

fig, ax = plt.subplots()
ax.pie(size_of_SubscriptionGroups1, radius=1, colors=colors,
       wedgeprops=dict(width=0.3, edgecolor='w'), autopct='%1.0f%%', pctdistance=1.13,
       textprops={'fontsize': 30, "fontweight": "bold"})
ax.pie(size_of_SubscriptionGroups2, radius=0.7, colors=colors,
       wedgeprops=dict(width=0.3, edgecolor='w'), autopct='%1.0f%%', pctdistance=0.8,
       textprops={'fontsize': 30,  "fontweight": "bold"})
plt.legend(labels, title="Outer ring = BSS trips  \nInner ring = BSS users \n\nUser's subscription\ntype",
           bbox_to_anchor=(1, 0.9), fontsize=24)
ax.get_legend().get_title().set_fontsize('24')
ax.set(aspect="equal")

plt.show()

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

b1 = sns.barplot(x='Month', y='id', data=MonthlyTripCountPerBike, ax=monthlyHist, color='B')
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
hourlyTripGroupByGender = hourlyTripGroupByGender.loc[
    (hourlyTripGroupByGender["hsl_gender"] == "male") | (hourlyTripGroupByGender["hsl_gender"] == "female")]

hourlyTripGroupByMen = hourlyTripGroupByGender.loc[hourlyTripGroupByGender["hsl_gender"] == "male"]
hourlyTripGroupByWomen = hourlyTripGroupByGender.loc[hourlyTripGroupByGender["hsl_gender"] == "female"]

hourlyTripGroupByGender["genderWeekday"] = hourlyTripGroupByGender["hsl_gender"] + "_" + hourlyTripGroupByGender[
    "WeekOrWeekend"].astype(str)
hourlyTripGroupByGender["tripPercent"] = hourlyTripGroupByGender.apply(
    lambda row: (row["index"] / hourlyTripGroupByMen["index"].sum()) * 100 if (row["hsl_gender"] == "male") else (
                                                                                                                             row[
                                                                                                                                 "index"] /
                                                                                                                             hourlyTripGroupByWomen[
                                                                                                                                 "index"].sum()) * 100,
    axis=1)

hourlyTripGroupByGender["Time of the week"] = hourlyTripGroupByGender.apply(lambda x: "weekday" if x["WeekOrWeekend"] == 1 else "weekend", axis = 1)
hourlyTripGroupByGender['Gender'] = hourlyTripGroupByGender['hsl_gender'].map(({"female": "Female", "male" : "Male" }))

deepColor = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
newPalette = dict(Female = deepColor[1], Male = deepColor[0])

ax = sns.lineplot(y="tripPercent", x="DepHour", hue="Gender", data=hourlyTripGroupByGender,
                  style="Time of the week  ", palette = newPalette, linewidth=5.0)
sns.set_context("poster", rc={"lines.linewidth": 5.5})
plt.xticks(np.arange(0, 24, step=1))
ax.set_xlabel('Departure hour', fontsize=30)
ax.set_ylabel('Share of total (%)', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text

plt.show()

# Lineplot 2: Hourly trip count variation by home area
hourlyTripGroupByHomeArea = manipulatedDF.groupby(["DepHour", "InsideArea", "WeekOrWeekend"]).count().reset_index()
hourlyTripGroupByHomeArea = hourlyTripGroupByHomeArea.loc[
    (hourlyTripGroupByHomeArea["InsideArea"] == 0) | (hourlyTripGroupByHomeArea["InsideArea"] == 1)]

hourlyTripGroupByInsideUsers = hourlyTripGroupByHomeArea.loc[hourlyTripGroupByHomeArea["InsideArea"] == 1]
hourlyTripGroupByOutsideUsers = hourlyTripGroupByHomeArea.loc[hourlyTripGroupByHomeArea["InsideArea"] == 0]

hourlyTripGroupByHomeArea["homeAreaWeekday"] = hourlyTripGroupByHomeArea["InsideArea"].astype(str) + "_" + \
                                               hourlyTripGroupByHomeArea["WeekOrWeekend"].astype(str)
hourlyTripGroupByHomeArea["tripPercent"] = hourlyTripGroupByHomeArea.apply(
    lambda row: (row["index"] / hourlyTripGroupByInsideUsers["index"].sum()) * 100
    if (row["InsideArea"] == 1)
    else (row["index"] / hourlyTripGroupByOutsideUsers["index"].sum()) * 100, axis=1)

hourlyTripGroupByHomeArea["Time of the week"] = hourlyTripGroupByHomeArea.apply(lambda x: "weekday" if x["WeekOrWeekend"] == 1 else "weekend", axis = 1)
hourlyTripGroupByHomeArea['Home area within \nBSS coverage area'] = hourlyTripGroupByHomeArea['InsideArea'].map(({0 : "No", 1: "Yes"}))

cmap = plt.get_cmap("tab20b")
colors = cmap(np.array([13,0]))

ax = sns.lineplot(y="tripPercent", x="DepHour", hue="Home area within \nBSS coverage area", data=hourlyTripGroupByHomeArea,
                  style="Time of the week", linewidth=5.0, palette = colors)
sns.set_context("poster", rc={"grid.linewidth": 0.6})
plt.xticks(np.arange(0, 24, step=1))
ax.set_xlabel('Departure hour', fontsize=30)
ax.set_ylabel('Share of total (%)', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text

plt.show()

# Lineplot 3: Hourly trip count variation by age group
manipulatedDF['age_dec'] = manipulatedDF.hsl_age.map(lambda hsl_age: 20 * (hsl_age // 20))  # Create age decade column

hourlyTripGroupByAgeGroup = manipulatedDF.groupby(["DepHour", "age_dec", "WeekOrWeekend"]).count().reset_index()
hourlyTripGroupByAgeGroup = hourlyTripGroupByAgeGroup.loc[hourlyTripGroupByAgeGroup["age_dec"] < 80]

hourlyTripGroupByAge_0_20 = hourlyTripGroupByAgeGroup.loc[hourlyTripGroupByAgeGroup["age_dec"] < 20]
hourlyTripGroupByAge_21_40 = hourlyTripGroupByAgeGroup.loc[
    (hourlyTripGroupByAgeGroup["age_dec"] >= 20) & (hourlyTripGroupByAgeGroup["age_dec"] < 40)]
hourlyTripGroupByAge_41_60 = hourlyTripGroupByAgeGroup.loc[
    (hourlyTripGroupByAgeGroup["age_dec"] >= 40) & (hourlyTripGroupByAgeGroup["age_dec"] < 60)]
hourlyTripGroupByAge_61_80 = hourlyTripGroupByAgeGroup.loc[
    (hourlyTripGroupByAgeGroup["age_dec"] >= 60) & (hourlyTripGroupByAgeGroup["age_dec"] < 80)]

hourlyTripGroupByAgeGroup["ageGroupWeekday"] = hourlyTripGroupByAgeGroup["age_dec"].astype(str) + "_" + \
                                               hourlyTripGroupByAgeGroup["WeekOrWeekend"].astype(str)
hourlyTripGroupByAgeGroup["tripPercent"] = hourlyTripGroupByAgeGroup.apply(
    lambda row: (row["index"] / hourlyTripGroupByAge_0_20["index"].sum()) * 100
    if (row["age_dec"] == 0)
    else ((row["index"] / hourlyTripGroupByAge_21_40["index"].sum()) * 100
          if row["age_dec"] == 20
          else ((row["index"] / hourlyTripGroupByAge_41_60["index"].sum()) * 100
                if row["age_dec"] == 40
                else ((row["index"] / hourlyTripGroupByAge_61_80["index"].sum()) * 100))), axis=1)

hourlyTripGroupByAgeGroup["Time of the week"] = hourlyTripGroupByAgeGroup.apply(lambda x: "weekday" if x["WeekOrWeekend"] == 1 else "weekend", axis = 1)
hourlyTripGroupByAgeGroup['Age group'] = hourlyTripGroupByAgeGroup['age_dec'].map(({0 : "0-19", 20: "20-39",40: "40-59", 60:"60-79"}))

ax = sns.lineplot(y="tripPercent", x="DepHour", hue="Age group", data=hourlyTripGroupByAgeGroup,
                   style="Time of the week", linewidth = 5.0)
sns.set_context("poster", rc={"grid.linewidth": 0.6})
plt.xticks(np.arange(0, 24, step=1))
ax.set_xlabel('Departure hour', fontsize=30)
ax.set_ylabel('Share of total (%)', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text

plt.show()

# Lineplot 4: Hourly trip count variation by subscription type
hourlyTripGroupByFormula = manipulatedAllUsersDF.groupby(["DepHour", "formula", "WeekOrWeekend"]).count().reset_index()
hourlyTripGroupByFormula = hourlyTripGroupByFormula.loc[
    (hourlyTripGroupByFormula["formula"] == "Day") | (hourlyTripGroupByFormula["formula"] == "Week") | (
            hourlyTripGroupByFormula["formula"] == "Year")]

hourlyTripGroupByDayUsers = hourlyTripGroupByFormula.loc[hourlyTripGroupByFormula["formula"] == "Day"]
hourlyTripGroupByWeekUsers = hourlyTripGroupByFormula.loc[hourlyTripGroupByFormula["formula"] == "Week"]
hourlyTripGroupByYearUsers = hourlyTripGroupByFormula.loc[hourlyTripGroupByFormula["formula"] == "Year"]

hourlyTripGroupByFormula["formulaWeekday"] = hourlyTripGroupByFormula["formula"].astype(str) + "_" + \
                                             hourlyTripGroupByFormula["WeekOrWeekend"].astype(str)
hourlyTripGroupByFormula["tripPercent"] = hourlyTripGroupByFormula.apply(
    lambda row: (row["index"] / hourlyTripGroupByDayUsers["index"].sum()) * 100
    if (row["formula"] == "Day")
    else ((row["index"] / hourlyTripGroupByWeekUsers["index"].sum()) * 100
          if row["formula"] == "Week"
          else ((row["index"] / hourlyTripGroupByYearUsers["index"].sum()) * 100)), axis=1)

hourlyTripGroupByFormula.rename(columns={"formula":"Subscription type"}, inplace=True)
hourlyTripGroupByFormula["Time of the week"] = hourlyTripGroupByFormula.apply(lambda x: "weekday" if x["WeekOrWeekend"] == 1 else "weekend", axis = 1)

cmap = plt.get_cmap("tab20b")
colors = cmap(np.array([10,1,4 ]))

ax = sns.lineplot(y="tripPercent", x="DepHour", hue="Subscription type", data=hourlyTripGroupByFormula,
                  style="Time of the week", linewidth = 5.0, palette = colors)
sns.set_context("poster", rc={"grid.linewidth": 0.6})
plt.xticks(np.arange(0, 24, step=1))
ax.set_xlabel('Departure hour', fontsize=30)
ax.set_ylabel('Share of total (%)', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text

plt.show()

# Lineplot 5: Hourly trip count variation by use activity
AllUsersCombinedUserGroup.sort_values('trip_count', ascending=False, inplace=True)
q = pd.qcut(AllUsersCombinedUserGroup["trip_count"], 5)
AllUsersCombinedUserGroup['ActivityQ'] = q

manipulatedAllUsersDF_merge = manipulatedAllUsersDF.merge(AllUsersCombinedUserGroup, left_on="uid", right_on="uid")

hourlyTripGroupByUseActivity = manipulatedAllUsersDF_merge.groupby(
    ["DepHour", "ActivityQ", "WeekOrWeekend"]).count().reset_index()

hourlyTripGroupByActivityQ1 = hourlyTripGroupByUseActivity.loc[
    hourlyTripGroupByUseActivity["ActivityQ"].astype(str) == "(0.999, 5.0]"]
hourlyTripGroupByActivityQ2 = hourlyTripGroupByUseActivity.loc[
    hourlyTripGroupByUseActivity["ActivityQ"].astype(str) == "(5.0, 13.0]"]
hourlyTripGroupByActivityQ3 = hourlyTripGroupByUseActivity.loc[
    hourlyTripGroupByUseActivity["ActivityQ"].astype(str) == "(13.0, 26.0]"]
hourlyTripGroupByActivityQ4 = hourlyTripGroupByUseActivity.loc[
    hourlyTripGroupByUseActivity["ActivityQ"].astype(str) == "(26.0, 55.0]"]
hourlyTripGroupByActivityQ5 = hourlyTripGroupByUseActivity.loc[
    hourlyTripGroupByUseActivity["ActivityQ"].astype(str) == "(55.0, 1124.0]"]


hourlyTripGroupByUseActivity["useActivityWeekday"] = hourlyTripGroupByUseActivity["ActivityQ"].astype(str) + "_" + \
                                                     hourlyTripGroupByUseActivity["WeekOrWeekend"].astype(str)

hourlyTripGroupByUseActivity["tripPercent"] = hourlyTripGroupByUseActivity.apply(
    lambda row: (row["index"] / hourlyTripGroupByActivityQ1["index"].sum()) * 100
    if str(row["ActivityQ"]) == "(0.999, 5.0]"
    else ((row["index"] / hourlyTripGroupByActivityQ2["index"].sum()) * 100
          if str(row["ActivityQ"]) == "(5.0, 13.0]"
          else ((row["index"] / hourlyTripGroupByActivityQ3["index"].sum()) * 100
                if str(row["ActivityQ"]) == "(13.0, 26.0]"
                else ((row["index"] / hourlyTripGroupByActivityQ4["index"].sum()) * 100
                      if str(row["ActivityQ"]) == "(26.0, 55.0]"
                      else ((row["index"] / hourlyTripGroupByActivityQ5["index"].sum()) * 100)))), axis=1)

hourlyTripGroupByUseActivity['Trip count by user \n(quintile)'] = hourlyTripGroupByUseActivity['ActivityQ'].astype(str).map(({"(0.999, 5.0]" : "1-4", "(5.0, 13.0]": "6-12","(13.0, 26.0]": "13-25", "(26.0, 55.0]":"26-54", "(55.0, 1124.0]" :"55-1124"}))
hourlyTripGroupByUseActivity["Time of the week"] = hourlyTripGroupByUseActivity.apply(lambda x: "weekday" if x["WeekOrWeekend"] == 1 else "weekend", axis = 1)

ax = sns.lineplot(y="tripPercent", x="DepHour", hue="Trip count by user \n(quintile)", data=hourlyTripGroupByUseActivity,
                  style="Time of the week", linewidth =5.0)
sns.set_context("poster", rc={"grid.linewidth": 0.6})
plt.xticks(np.arange(0, 24, step=1))
ax.set_xlabel('Departure hour', fontsize=30)
ax.set_ylabel('Share of total (%)', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.setp(ax.get_legend().get_texts(), fontsize='25') # for legend text
plt.show()

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
AllUsersCombinedUserGroup['trip_count'].describe()
AllUsersCombinedUserGroup['tripsPerDay'].describe()
AllUsersCombinedUserGroup['DayOfTheYear_nunique'].describe()

fig = plt.figure(figsize=(12, 3))
grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.2)

tripCountHist = fig.add_subplot(grid[0, 0])
tripCountHist.hist(AllUsersCombinedUserGroup['trip_count'], range=(0, 400), bins=20)
tripCountHist.set_xlabel('Trip count', fontsize=35)
tripCountHist.set_ylabel('User count', fontsize=35)
tripCountHist.tick_params(axis='x', which='major', labelsize=30)
tripCountHist.text(160, 4000, "mean: 34.8 \nstd: 44.9 \nmin: 1 \n25%: 6 \n50%: 18 \n75% 46 \nmax: 1124", size=25,
                   fontname="Verdana", weight="bold")

tripsPerDayHist = fig.add_subplot(grid[0, 1])
tripsPerDayHist.hist(AllUsersCombinedUserGroup['DayOfTheYear_nunique'], range=(0, 200), bins=20, color="darkslategray")
tripsPerDayHist.set_xlabel('Unique user days', fontsize=35)
tripsPerDayHist.set_ylabel('User count', fontsize=35)
tripsPerDayHist.tick_params(axis='x', which='both', labelsize=30)
tripsPerDayHist.text(75, 3500, "mean: 20.1 \nstd: 22.6 \nmin: 1 \n25%: 3 \n50%: 12 \n75% 29 \nmax: 158", size=25,
                     fontname="Verdana", weight="bold")

tripsPerDayHist = fig.add_subplot(grid[0, 2])
tripsPerDayHist.hist(AllUsersCombinedUserGroup['tripsPerDay'], range=(0, 3), bins=20, color="indianred")
tripsPerDayHist.set_xlabel('Trips per day ', fontsize=25)
tripsPerDayHist.set_ylabel('User count', fontsize=25)
tripsPerDayHist.text(2.0, 5000, "mean: 0.20 \nstd: 0.26 \nmin: 0.005 \n25%: 0.03 \n50%: 0.10 \n75% 0.26 \nmax: 6.42",
                     size=15, fontname="Verdana", weight="bold")

AllUsersCombinedUserGroup['PT_trip_pros'].hist(bins=20)
AllUsersCombinedUserGroup['userDayRatio'].hist(bins=20)
AllUsersCombinedUserGroup['userDayCount'].hist(bins=20)

# Cumulative trip count plot: 
AllUsersCombinedUserGroup.sort_values('trip_count', ascending=False, inplace=True)
AllUsersCombinedUserGroup['cum_sum'] = AllUsersCombinedUserGroup.trip_count.cumsum()
AllUsersCombinedUserGroup['cum_perc'] = 100 * AllUsersCombinedUserGroup.cum_sum / AllUsersCombinedUserGroup.trip_count.sum()

sns.set(font_scale=2.1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Cumulative percentage of users % ', fontsize=35)
ax.set_ylabel('Cumulative percentage of trips %', fontsize=35)

AllUsersCombinedUserGroup = AllUsersCombinedUserGroup.sort_values('cum_sum', ascending=True)
AllUsersCombinedUserGroup['seq'] = range(1, len(AllUsersCombinedUserGroup) + 1)
AllUsersCombinedUserGroup['user_perc'] = 100 * AllUsersCombinedUserGroup.seq / len(AllUsersCombinedUserGroup.seq)

ax.plot(AllUsersCombinedUserGroup['user_perc'], AllUsersCombinedUserGroup['cum_perc'], linewidth = 6.0)
ax.tick_params(axis='both', which='major', labelsize=30)
percentage = 0.2
while percentage < 1.1:
    point = int(round(len(AllUsersCombinedUserGroup) * percentage, 0))
    ax.text(AllUsersCombinedUserGroup.iloc[point]["user_perc"], AllUsersCombinedUserGroup.iloc[point]["cum_perc"],
            "User count: \n %s \n Trip count:\n %s" % (
                AllUsersCombinedUserGroup.iloc[point]["seq"], AllUsersCombinedUserGroup.iloc[point]["cum_sum"]),
            horizontalalignment='left',
            verticalalignment='top',
            multialignment='center',
            fontsize=34)
    ax1 = fig.add_subplot(111)
    ax1.plot([AllUsersCombinedUserGroup.iloc[point]["user_perc"]], [AllUsersCombinedUserGroup.iloc[point]["cum_perc"]], marker='o',
             markersize=14, markerfacecolor="darkred")
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
    (CombinedUserGroup['hsl_age'] > 0) & (CombinedUserGroup['hsl_age'] < 80)]
FilteredCombinedUserGroup['age_dec'] = CombinedUserGroup.hsl_age.map(
    lambda hsl_age: 20 * (hsl_age // 20))  # create the age by decade column

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
CombinedUserGroup.sort_values('trip_count', ascending=False, inplace=True)
q = pd.qcut(CombinedUserGroup["trip_count"], 5)
CombinedUserGroup['ActivityQ'] = q

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
