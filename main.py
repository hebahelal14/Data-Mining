import inline as inline
import matplotlib
import pandas as pd
import pdpipe as pdp
import numpy as np
from Tools.scripts.dutree import display
from sklearn import preprocessing
import time
from datetime import datetime
import nltk
from google.cloud import bigquery
import pycountry
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from pandas.io import gbq

#getting the covid-19 Datasets
confirmed_ts_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
deaths_ts_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
recovered_ts_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")

#print(confirmed_ts_df)
#print(deaths_ts_df)
#print(recovered_ts_df)


#Covert Data from wide to long based on the data attribute (melt is function in pandas)
confirmed_ts_melted_df = confirmed_ts_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                     var_name='covid_date', value_name='confirmed').copy()

deaths_ts_melted_df = deaths_ts_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                     var_name='covid_date', value_name='deaths').copy()

recovered_ts_melted_df = recovered_ts_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                     var_name='covid_date', value_name='recovered').copy()

#confirmed_ts_melted_df.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\confirmed_ts_melted_df.csv')

#sort data according to the country and date
confirmed_ts_melted_df = confirmed_ts_melted_df.sort_values(by=['Country/Region', 'covid_date'])
deaths_ts_melted_df = deaths_ts_melted_df.sort_values(by=['Country/Region', 'covid_date'])
recovered_ts_melted_df = recovered_ts_melted_df.sort_values(by=['Country/Region', 'covid_date'])

#sort according to the index
confirmed_ts_fcg_df = confirmed_ts_melted_df.sort_index(axis = 0)
deaths_ts_fcg_df = deaths_ts_melted_df.sort_index(axis = 0)
recovered_ts_fcg_df = recovered_ts_melted_df.sort_index(axis = 0)

#merge all 3 DataFrame into one DataFrame
covid_df = pd.merge(confirmed_ts_melted_df, deaths_ts_melted_df,on=['Country/Region','Province/State','covid_date','Lat','Long'])
covid_df2 = pd.merge(covid_df, recovered_ts_melted_df,on=['Country/Region','Province/State','covid_date'])
#covid_df.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\covid_df.csv')

#select only Gulf countries, China, Candinavian Countries, US, Canada, Brazil, and India
covid_df = covid_df[(covid_df['Country/Region'] == 'United Arab Emirates') | (covid_df['Country/Region'] == 'Bahrain') | (covid_df['Country/Region'] == 'Qatar') | (covid_df['Country/Region'] == 'Oman') | (covid_df['Country/Region'] == 'Saudi Arabia') | (covid_df['Country/Region'] == 'Kuwait') | (covid_df['Country/Region'] == 'China')| (covid_df['Country/Region'] == 'US')| (covid_df['Country/Region'] == 'Sweden')|(covid_df['Country/Region'] == 'Denmark')|(covid_df['Country/Region'] == 'Norway')|(covid_df['Country/Region'] == 'Iceland')|(covid_df['Country/Region'] == 'Finland')|(covid_df['Country/Region'] == 'Canada')|(covid_df['Country/Region'] == 'Brazil')|(covid_df['Country/Region'] == 'India')]

#check the data-time
covid_df['covid_date'] = pd.to_datetime(covid_df['covid_date'], errors='coerce')

#sort data according to the country and date
covid_df = covid_df.sort_values(by=['Country/Region', 'covid_date'])

#save covid_df into file
##covid_df.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\covid_df.csv')

#Getting world population
sql = """
SELECT 
country_code as country_id, -- alpha3 country code
year_2018 as pop
FROM `bigquery-public-data.world_bank_global_population.population_by_country` 
WHERE year_2018 IS NOT NULL
"""

client = bigquery.Client()

# Set up the query
query_job = client.query(sql)

# Make an API request  to run the query and return a pandas DataFrame
pop_df = query_job.to_dataframe()

# map country alpha 3 codes to country names to display in the map
def get_country_name_by_alpha3(alpha_3_code):
    '''
    Takes the alpha-3 country code as an input
    Returns a full country name
    '''
    country_name = ""
    country_obj = pycountry.countries.get(alpha_3=alpha_3_code)
    if country_obj is None:
        country_name = alpha_3_code
    else:
        country_name = country_obj.name
    return country_name

country_name_pipeline3 = pdp.PdPipeline([
    pdp.ApplyByCols(['country_id'], get_country_name_by_alpha3),
])
# replace country alpha3 code with country name
pop_df = country_name_pipeline3.apply(pop_df)

#Rename Column country_id with Country/Region
pop_df.rename (columns= { 'country_id': 'Country/Region' }, inplace= True)

#select only Gulf countries, China, Sweden, and US
pop_df = pop_df[(pop_df['Country/Region'] == 'United Arab Emirates') | (pop_df['Country/Region'] == 'Bahrain') | (pop_df['Country/Region'] == 'Qatar') | (pop_df['Country/Region'] == 'Oman') | (pop_df['Country/Region'] == 'Saudi Arabia') | (pop_df['Country/Region'] == 'Kuwait') | (pop_df['Country/Region'] == 'China')| (pop_df['Country/Region'] == 'United States')| (pop_df['Country/Region'] == 'Sweden')|(pop_df['Country/Region'] == 'Denmark')|(pop_df['Country/Region'] == 'Finland')|(pop_df['Country/Region'] == 'Iceland')|(pop_df['Country/Region'] == 'Norway')|(pop_df['Country/Region'] == 'Canada')|(pop_df['Country/Region'] == 'India')|(pop_df['Country/Region'] == 'Brazil')]
#pop_df.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\pop_df.csv')

#Change from United State to US
pop_df['Country/Region'] = ['United Arab Emirates','Saudi Arabia','Qatar','US','China','Oman','Bahrain','Kuwait','Sweden','Denmark','Finland','Iceland','Norway','Canada','India','Brazil']

#save pop_df into file
##pop_df.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\pop_df.csv')

# merge COVID-19 and population datasets
covid_df = pd.merge(covid_df,pop_df,on='Country/Region')
#covid_df.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\covid_df_total.csv')

# Check Missing value summary
nan_columns = []
nan_values = []

for column in covid_df.columns:
    nan_columns.append(column)
    nan_values.append(covid_df[column].isnull().sum())

#Data Preprocessing
#Convert date from string to numeric value which DM can deal with it
le = preprocessing.LabelEncoder()

covid_df_corr = covid_df.copy()
covid_df_corr['day_num'] = le.fit_transform(covid_df_corr.covid_date)
covid_df_corr['day_num'] = pd.to_datetime(covid_df_corr['covid_date'], errors='coerce')
covid_df_corr['day'] = covid_df_corr['day_num'].dt.day
covid_df_corr['month'] = covid_df_corr['day_num'].dt.month
covid_df_corr['year'] = covid_df_corr['day_num'].dt.year
##covid_df_corr.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\covid_df_corr.csv')

#Adding Additional Pandemic Features and Country Population
def add_extra_trends(df):
    df1 = df.copy()
    df1['case_fatality_rate'] = df1['deaths'] / df1['confirmed']
    df1['infection_rate'] = df1['confirmed'] / df1['pop']
    df1['mortality_rate'] = df1['deaths'] / df1['pop']
    df1['case_fatality_rate'] = round(df1['case_fatality_rate'].fillna(0), 4)
    #df1['infection_rate'] = round(df1['infection_rate'].fillna(0), 4)
    #df1['mortality_rate'] = round(df1['mortality_rate'].fillna(0), 4)
    return df1
covid_df_corr = add_extra_trends(covid_df_corr)
#covid_df_corr.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\covid_df_corr_extra.csv')

#Correlation between attributes in COVID-19 pandemic
corr_transform = covid_df_corr.drop(['Province/State', 'Country/Region','covid_date','day','month','year','day_num'], axis=1)
#print(corr_transform)
#corr_transform.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\covid_df_corr_heatmap.csv')
#default correlation is Pearson
corr = corr_transform.corr()

ax = sns.heatmap(corr,annot = True,cmap='Blues',square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

ax.set_title('Correlation of different attributes in COVID-19 ')
plt.show()

def heatmap_numeric_w_dependent_variable(df, dependent_variable):

    plt.figure(figsize=(8, 10))
    g = sns.heatmap(df[[dependent_variable]].sort_values(by=dependent_variable),
                    annot=True,
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1)
    g.set_yticklabels(
        g.get_yticklabels(),
        rotation=45,
        horizontalalignment='right')
    print(df)
    return g

#Get Correlations with Confirmed
heatmap_numeric_w_dependent_variable(corr, 'confirmed')
#Get Correlations with Deaths
#heatmap_numeric_w_dependent_variable(corr, 'deaths')
#Get Correlations with Recovered
#heatmap_numeric_w_dependent_variable(covid_df_corr, 'recovered')
plt.show()

# map country alpha 2 codes to country names to display in the map
def get_country_name(alpha_2_code):
    '''
    Takes the alpha-2 country code as an input
    Returns a full country name
    '''
    country_obj = pycountry.countries.get(alpha_2=alpha_2_code)
    country_name = country_obj.name
    return country_name

country_name_pipeline2 = pdp.PdPipeline([
    pdp.ApplyByCols(['country_id'], get_country_name),
])
# we are going to extract Google_mobility datasets
#mobility_df = pd.read_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\google_mobility.csv')

# we are going to extract data from BigQuery
# SQL Query for aggregated mobility trend data
sql = """
  SELECT
  country_region_code as country_id, 
  date as covid_date,
  ROUND(AVG(retail_and_recreation_percent_change_from_baseline), 4) as retail_and_recreation_percent_change_from_baseline, 
  ROUND(AVG(grocery_and_pharmacy_percent_change_from_baseline), 4) as grocery_and_pharmacy_percent_change_from_baseline,
  ROUND(AVG(parks_percent_change_from_baseline), 4) as parks_percent_change_from_baseline,
  ROUND(AVG(transit_stations_percent_change_from_baseline), 4) as transit_stations_percent_change_from_baseline,
  ROUND(AVG(workplaces_percent_change_from_baseline), 4) as workplaces_percent_change_from_baseline,
  ROUND(AVG(residential_percent_change_from_baseline), 4) as residential_percent_change_from_baseline
  FROM `bigquery-public-data.covid19_google_mobility.mobility_report`
  GROUP BY country_region_code, date
"""
# Set up the query
query_job = client.query(sql)
# Make an API request  to run the query and return a pandas DataFrame
mobility_df = query_job.to_dataframe()

# replace country alpha2 code with country name
mobility_df = country_name_pipeline2.apply(mobility_df)
mobility_df.rename (columns= { 'date': 'covid_date' }, inplace= True)
mobility_df.rename (columns= { 'country_id': 'Country/Region' }, inplace= True)
#mobility_df.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\mobility.csv')

#select only Gulf countries, China, Candinavian Countries, US, India, and Brazil
mobility_df = mobility_df[(mobility_df['Country/Region'] == 'United Arab Emirates') | (mobility_df['Country/Region'] == 'Bahrain') | (mobility_df['Country/Region'] == 'Qatar') | (mobility_df['Country/Region'] == 'Oman') | (mobility_df['Country/Region'] == 'Saudi Arabia') | (mobility_df['Country/Region'] == 'Kuwait') | (mobility_df['Country/Region'] == 'China')| (mobility_df['Country/Region'] == 'United States')| (mobility_df['Country/Region'] == 'Sweden')|(mobility_df['Country/Region'] == 'Denmark')|(mobility_df['Country/Region'] == 'Finland')|(mobility_df['Country/Region'] == 'Iceland')|(mobility_df['Country/Region'] == 'Norway')|(mobility_df['Country/Region'] == 'Canada')|(mobility_df['Country/Region'] == 'India')|(mobility_df['Country/Region'] == 'Brazil')]

#Change from United State to US
mobility_df['Country/Region'] = mobility_df['Country/Region'].replace(['United States'],'US')

#sort data according to the country and date
mobility_df = mobility_df.sort_values(by=['Country/Region', 'covid_date'])
#save google_mobility_df into csv file
#mobility_df.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\mobility_df.csv')

# Check Missing value summary
nan_columns = []
nan_values = []

for column in mobility_df.columns:
    nan_columns.append(column)
    nan_values.append(mobility_df[column].isnull().sum())
#print(nan_columns)
#print(nan_values)

#Preprocessing Data
#Convert date from string to numeric value which DM can deal with it
le = preprocessing.LabelEncoder()

mobility_df_corr = mobility_df.copy()
mobility_df_corr['day_num'] = le.fit_transform(mobility_df_corr.covid_date)
mobility_df_corr['day_num'] = pd.to_datetime(mobility_df_corr['covid_date'], errors='coerce')
mobility_df_corr['day'] = mobility_df_corr['day_num'].dt.day
#print(mobility_df_corr['day_num'])
mobility_df_corr['month'] = mobility_df_corr['day_num'].dt.month
mobility_df_corr['year'] = mobility_df_corr['day_num'].dt.year
mobility_df_corr = mobility_df_corr.sort_values(by=['Country/Region', 'covid_date'])
mobility_df_corr.reset_index(drop=True, inplace=True)
#mobility_df_corr.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\clustering\\mobility_df.csv')

# Correlation between attributes in Google_mobility
corr_transform1 = mobility_df_corr.drop([ 'Country/Region','covid_date','day','month','year','day_num'], axis=1)

#print(corr_transform)
#corr_transform.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\covid_df_corr_heatmap.csv')

#default correlation is Pearson
##comment from corr1 to ax.set_title
corr1 = corr_transform1.corr()
ax = sns.heatmap(corr1,annot = True,cmap='Blues',square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
ax.set_title('Correlation of different attributes in Google_mobility ')
plt.show()

print('------------------------------------------TOTAL--------------------------------------------------')
#Merge COVID-19 Pandemic Attirbutes and Google Mobility Trend Features
total = pd.merge(covid_df_corr,mobility_df_corr,on=['Country/Region', 'day_num'])

#drop Repeated columns
total.drop(['covid_date_y','day_y','month_y','year_y'], axis=1,inplace= True)
#total.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\total.csv')

# Correlation between attributes in COVID-19 pandemic and Google Mobility Trend Features
retail = total.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)

#print(corr_transform)
#corr_transform.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\covid_df_corr_heatmap.csv')

#default correlation is Pearson
Retail = retail.corr()
#print(retail)
heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
grocery =total.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
park =total.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
transit =total.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
workplace =total.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
residential =total.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
plt.show()

##select country Bahrain from total dataset
Bahrain = total.loc[total['Country/Region']=='Bahrain']
#print(Bahrain)
##Bahrain.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Bahrain.csv')
##select country Kuwait from total dataset
Kuwait = total.loc[total['Country/Region']=='Kuwait']
Kuwait.reset_index(drop=True, inplace=True)
##Kuwait.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Kuwait.csv')
##select country Oman from total dataset
Oman = total.loc[total['Country/Region']=='Oman']
Oman.reset_index(drop=True, inplace=True)
##Oman.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Oman.csv')
##select country Qatar from total dataset
Qatar = total.loc[total['Country/Region']=='Qatar']
Qatar.reset_index(drop=True, inplace=True)
##Qatar.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Qatar.csv')
##select country Saudi Arabia from total dataset
Saudi_Arabia = total.loc[total['Country/Region']=='Saudi Arabia']
Saudi_Arabia.reset_index(drop=True, inplace=True)
##Saudi_Arabia.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Saudi_Arabia.csv')
##select country Sweden from total dataset
Sweden = total.loc[total['Country/Region']=='Sweden']
Sweden.reset_index(drop=True, inplace=True)
##Sweden.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Sweden.csv')
##select country US from total dataset
US = total.loc[total['Country/Region']=='US']
US.reset_index(drop=True, inplace=True)
##US.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\US.csv')
##select country UAE from total dataset
UAE = total.loc[total['Country/Region']=='United Arab Emirates']
UAE.reset_index(drop=True, inplace=True)
##UAE.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\UAE.csv')
##select country Denmark from total dataset
Denmark = total.loc[total['Country/Region']=='Denmark']
Denmark.reset_index(drop=True, inplace=True)
#Denmark.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Denmark.csv')
##select country Finland from total dataset
Finland = total.loc[total['Country/Region']=='Finland']
Finland.reset_index(drop=True, inplace=True)
#Finland.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Finland.csv')
##select country Norway from total dataset
Norway = total.loc[total['Country/Region']=='Norway']
Norway.reset_index(drop=True, inplace=True)
#Norway.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Norway.csv')
##select country Canada from total dataset
Canada = total.loc[total['Country/Region']=='Canada']
Canada.reset_index(drop=True, inplace=True)
#Canada.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Canada.csv')
##select country India from total dataset
India = total.loc[total['Country/Region']=='India']
India.reset_index(drop=True, inplace=True)
#India.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\India.csv')
##select country Brazil from total dataset
Brazil = total.loc[total['Country/Region']=='Brazil']
Brazil.reset_index(drop=True, inplace=True)
#Brazil.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Countries_Dataset\\Brazil.csv')


#Make Correlation for Bahrain using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Bahrain.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#print(Retail)
heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
grocery =Bahrain.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
park =Bahrain.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
transit =Bahrain.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
workplace =Bahrain.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
residential =Bahrain.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
plt.show()

#Make Correlation for Kuwait using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Kuwait.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
grocery =Kuwait.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
park =Kuwait.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
transit =Kuwait.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
workplace =Kuwait.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
residential =Kuwait.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
plt.show()

#Make Correlation for Oman using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Oman.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
grocery =Oman.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
park =Oman.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
transit =Oman.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
plt.show()

#default correlation is Pearson
workplace =Oman.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =Oman.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()


#Make Correlation for Qatar using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Qatar.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =Qatar.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =Qatar.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =Qatar.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
#Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =Qatar.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =Qatar.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()



#Make Correlation for Saudi_Arabia using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Saudi_Arabia.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =Saudi_Arabia.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =Saudi_Arabia.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =Saudi_Arabia.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =Saudi_Arabia.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =Saudi_Arabia.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()


#Make Correlation for Sweden using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Sweden.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =Sweden.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =Sweden.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =Sweden.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =Sweden.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =Sweden.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()


#Make Correlation for US using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = US.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =US.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =US.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =US.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =US.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =US.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()

#Make Correlation for UAE using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = UAE.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =UAE.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =UAE.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =UAE.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =UAE.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =UAE.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()

#Make Correlation for Denmark using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Denmark.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =Denmark.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =Denmark.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =Denmark.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =Denmark.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =Denmark.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()


#Make Correlation for Finland using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Finland.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =Finland.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =Finland.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =Finland.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =Finland.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =Finland.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()

#Make Correlation for Norway using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Norway.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =Norway.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =Norway.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =Norway.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =Norway.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =Norway.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()

#Make Correlation for Canada using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Canada.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =Canada.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =Canada.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =Canada.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =Canada.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =Canada.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()

#Make Correlation for Brazil using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = Brazil.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =Brazil.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =Brazil.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =Brazil.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =Brazil.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =Brazil.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()

#Make Correlation for India using Pearson & Kendal & Spearman
#default correlation is Pearson
retail = India.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Retail = retail.corr()
#heatmap_numeric_w_dependent_variable(Retail, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Kendal
Retail_kendall = retail.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Retail_kendall, 'retail_and_recreation_percent_change_from_baseline')
##correlation using Spearman
Retail_spearman = retail.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Retail_spearman, 'retail_and_recreation_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
grocery =India.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Grocery = grocery.corr()
#heatmap_numeric_w_dependent_variable(Grocery, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Kendal
Grocery_kendall = grocery.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Grocery_kendall, 'grocery_and_pharmacy_percent_change_from_baseline')
##correlation using Spearman
Grocery_spearman = grocery.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Grocery_spearman, 'grocery_and_pharmacy_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
park =India.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Park = park.corr()
#heatmap_numeric_w_dependent_variable(Park,'parks_percent_change_from_baseline')
##correlation using Kendal
Park_kendall = park.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Park_kendall,'parks_percent_change_from_baseline')
##correlation using Spearman
Park_spearman = park.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Park_spearman,'parks_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
transit =India.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Transit = transit.corr()
#heatmap_numeric_w_dependent_variable(Transit,'transit_stations_percent_change_from_baseline')
##correlation using Kendal
Transit_kendall = transit.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Transit_kendall,'transit_stations_percent_change_from_baseline')
##correlation using Spearman
Transit_spearman = transit.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Transit_spearman,'transit_stations_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
workplace =India.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','residential_percent_change_from_baseline'], axis=1)
Workplace = workplace.corr()
#heatmap_numeric_w_dependent_variable(Workplace,'workplaces_percent_change_from_baseline')
##correlation using Kendal
Workplace_kendall = workplace.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Workplace_kendall,'workplaces_percent_change_from_baseline')
##correlation using Spearman
Workplace_spearman = workplace.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Workplace_spearman,'workplaces_percent_change_from_baseline')
#plt.show()

#default correlation is Pearson
residential =India.drop(['Province/State', 'Country/Region','covid_date_x','day_x','month_x','year_x','day_num','retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline'], axis=1)
Residential = residential.corr()
#heatmap_numeric_w_dependent_variable(Residential, 'residential_percent_change_from_baseline')
##correlation using Kendal
Residential_kendall = residential.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Residential_kendall, 'residential_percent_change_from_baseline')
##correlation using Spearman
Residential_spearman = residential.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Residential_spearman, 'residential_percent_change_from_baseline')
#plt.show()

print("--------------------------------------------------WEATHER DATASET---------------------------------------")
#Get weather Datasets from NOAA in Weather-dataset.py

weather_df=pd.read_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-dataset\\weather_dataset.csv')
#print(weather_df)
#rename
weather_df.rename (columns= { 'Date': 'covid_date' }, inplace= True)
weather_df.rename (columns= { 'Country_Region': 'Country/Region' }, inplace= True)
weather_df.rename (columns= { 'Province_State': 'Province/State' }, inplace= True)
weather_df.rename (columns= { 'ConfirmedCases': 'confirmed' }, inplace= True)
weather_df.rename (columns= { 'Fatalities': 'deaths' }, inplace= True)

# Check Missing value summary
nan_columns = []
nan_values = []

for column in mobility_df.columns:
    nan_columns.append(column)
    nan_values.append(mobility_df[column].isnull().sum())
#print(nan_columns)
#print(nan_values)

#select only Gulf countries, China, Sweden, and US
#weather_df = weather_df[(weather_df['Country/Region'] == 'United Arab Emirates') | (weather_df['Country/Region'] == 'Bahrain') | (weather_df['Country/Region'] == 'Qatar') | (weather_df['Country/Region'] == 'Oman') | (weather_df['Country/Region'] == 'Saudi Arabia') | (weather_df['Country/Region'] == 'Kuwait') | (weather_df['Country/Region'] == 'China')| (weather_df['Country/Region'] == 'US')| (weather_df['Country/Region'] == 'Sweden')]
weather_df = weather_df[(weather_df['Country/Region'] == 'United Arab Emirates') | (weather_df['Country/Region'] == 'Bahrain') | (weather_df['Country/Region'] == 'Qatar') | (weather_df['Country/Region'] == 'Oman') | (weather_df['Country/Region'] == 'Saudi Arabia') | (weather_df['Country/Region'] == 'Kuwait') | (weather_df['Country/Region'] == 'China')| (weather_df['Country/Region'] == 'US')| (weather_df['Country/Region'] == 'Sweden')|(weather_df['Country/Region'] == 'Denmark')|(weather_df['Country/Region'] == 'Norway')|(weather_df['Country/Region'] == 'Iceland')|(weather_df['Country/Region'] == 'Finland')|(weather_df['Country/Region'] == 'Canada')|(weather_df['Country/Region'] == 'Brazil')|(weather_df['Country/Region'] == 'India')]

weather_df = weather_df.drop(['country+province','day_from_jan_first','Id'], axis=1)
weather_df = weather_df.sort_values(by=['Country/Region', 'covid_date'])
weather_df.reset_index(drop=True, inplace=True)

# Fill null values given that we merged train-test datasets
weather_df['Province/State'].fillna("None", inplace=True)
weather_df['slp'].fillna(0, inplace=True)
#weather_df.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\weather_df+scandinavian.csv')

# Check Missing value summary
nan_columns = []
nan_values = []

for column in mobility_df.columns:
    nan_columns.append(column)
    nan_values.append(mobility_df[column].isnull().sum())
#print(nan_columns)
#print(nan_values)


##select country Bahrain from total dataset
Bahrain = weather_df.loc[weather_df['Country/Region']=='Bahrain']
#print(Bahrain)
##Bahrain.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Bahrain.csv')
##select country Kuwait from total dataset
Kuwait = weather_df.loc[weather_df['Country/Region']=='Kuwait']
Kuwait.reset_index(drop=True, inplace=True)
##Kuwait.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Kuwait.csv')
##select country Oman from total dataset
Oman = weather_df.loc[weather_df['Country/Region']=='Oman']
Oman.reset_index(drop=True, inplace=True)
##Oman.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Oman.csv')
##select country Qatar from total dataset
Qatar = weather_df.loc[weather_df['Country/Region']=='Qatar']
Qatar.reset_index(drop=True, inplace=True)
##Qatar.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Qatar.csv')
##select country Saudi Arabia from total dataset
Saudi_Arabia = weather_df.loc[weather_df['Country/Region']=='Saudi Arabia']
Saudi_Arabia.reset_index(drop=True, inplace=True)
##Saudi_Arabia.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Saudi_Arabia.csv')
##select country Sweden from total dataset
Sweden = weather_df.loc[weather_df['Country/Region']=='Sweden']
Sweden.reset_index(drop=True, inplace=True)
##Sweden.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Sweden.csv')
##select country US from total dataset
US = weather_df.loc[weather_df['Country/Region']=='US']
US.reset_index(drop=True, inplace=True)
##US.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\US.csv')
##select country UAE from total dataset
UAE = weather_df.loc[weather_df['Country/Region']=='United Arab Emirates']
UAE.reset_index(drop=True, inplace=True)
##UAE.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\UAE.csv')
##select country China from total dataset
China = weather_df.loc[weather_df['Country/Region']=='China']
China.reset_index(drop=True, inplace=True)
##China.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\China.csv')
##select country Denmark from total dataset
Denmark = weather_df.loc[weather_df['Country/Region']=='Denmark']
Denmark.reset_index(drop=True, inplace=True)
#Denmark.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Denmark.csv')
##select country Norway from total dataset
Norway = weather_df.loc[weather_df['Country/Region']=='Norway']
Norway.reset_index(drop=True, inplace=True)
#Norway.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Norway.csv')
##select country Iceland from total dataset
Iceland = weather_df.loc[weather_df['Country/Region']=='Iceland']
Iceland.reset_index(drop=True, inplace=True)
#Iceland.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Iceland.csv')
##select country Finland from total dataset
Finland = weather_df.loc[weather_df['Country/Region']=='Finland']
Finland.reset_index(drop=True, inplace=True)
#Finland.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Finland.csv')
##select country Canada from total dataset
Canada = weather_df.loc[weather_df['Country/Region']=='Canada']
Canada.reset_index(drop=True, inplace=True)
#Canada.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Canada.csv')
##select country Brazil from total dataset
Brazil = weather_df.loc[weather_df['Country/Region']=='Brazil']
Brazil.reset_index(drop=True, inplace=True)
#Brazil.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\Brazil.csv')
##select country India from total dataset
India = weather_df.loc[weather_df['Country/Region']=='India']
India.reset_index(drop=True, inplace=True)
#India.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Weather-Countries-Datasets\\India.csv')


#Make Correlation for Bahrain using Pearson & Kendal & Spearman
#default correlation is Pearson
Bahrain1 = Bahrain.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Bahrain_pearson = Bahrain1.corr()
#heatmap_numeric_w_dependent_variable(Bahrain_pearson, 'confirmed')
##correlation using Kendal
Bahrain_kendall = Bahrain1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Bahrain_kendall, 'confirmed')
##correlation using Spearman
Bahrain_spearman = Bahrain1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Bahrain_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Bahrain2 = Bahrain.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Bahrain_pearson = Bahrain2.corr()
#heatmap_numeric_w_dependent_variable(Bahrain_pearson, 'deaths')
##correlation using Kendal
Bahrain_kendall = Bahrain2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Bahrain_kendall, 'deaths')
##correlation using Spearman
Bahrain_spearman = Bahrain2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Bahrain_spearman, 'deaths')
#plt.show()

#Make Correlation for Kuwait using Pearson & Kendal & Spearman
#default correlation is Pearson
Kuwait1 = Kuwait.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Kuwait_pearson = Kuwait1.corr()
#heatmap_numeric_w_dependent_variable(Kuwait_pearson, 'confirmed')
##correlation using Kendal
Kuwait_kendall = Kuwait1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Kuwait_kendall, 'confirmed')
##correlation using Spearman
Kuwait_spearman = Kuwait1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Kuwait_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Kuwait2 = Kuwait.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Kuwait_pearson = Kuwait2.corr()
#heatmap_numeric_w_dependent_variable(Kuwait_pearson, 'deaths')
##correlation using Kendal
Kuwait_kendall = Kuwait2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Kuwait_kendall, 'deaths')
##correlation using Spearman
Kuwait_spearman = Kuwait2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Kuwait_spearman, 'deaths')
#plt.show()

#Make Correlation for Oman using Pearson & Kendal & Spearman
#default correlation is Pearson
Oman1 = Oman.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Oman_pearson = Oman1.corr()
#heatmap_numeric_w_dependent_variable(Oman_pearson, 'confirmed')
##correlation using Kendal
Oman_kendall = Oman1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Oman_kendall, 'confirmed')
##correlation using Spearman
Oman_spearman = Oman1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Oman_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Oman2 = Oman.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Oman_pearson = Oman2.corr()
#heatmap_numeric_w_dependent_variable(Oman_pearson, 'deaths')
##correlation using Kendal
Oman_kendall = Oman2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Oman_kendall, 'deaths')
##correlation using Spearman
Oman_spearman = Oman2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Oman_spearman, 'deaths')
#plt.show()


#Make Correlation for Qatar using Pearson & Kendal & Spearman
#default correlation is Pearson
Qatar1 = Qatar.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Qatar_pearson = Qatar1.corr()
#heatmap_numeric_w_dependent_variable(Qatar_pearson, 'confirmed')
##correlation using Kendal
Qatar_kendall = Qatar1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Qatar_kendall, 'confirmed')
##correlation using Spearman
Qatar_spearman = Qatar1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Qatar_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Qatar2 = Qatar.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Qatar_pearson = Qatar2.corr()
#heatmap_numeric_w_dependent_variable(Qatar_pearson, 'deaths')
##correlation using Kendal
Qatar_kendall = Qatar2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Qatar_kendall, 'deaths')
##correlation using Spearman
Qatar_spearman = Qatar2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Qatar_spearman, 'deaths')
#plt.show()



#Make Correlation for Saudi_Arabia using Pearson & Kendal & Spearman
#default correlation is Pearson
Saudi_Arabia1 = Saudi_Arabia.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Saudi_Arabia_pearson = Saudi_Arabia1.corr()
#heatmap_numeric_w_dependent_variable(Saudi_Arabia_pearson, 'confirmed')
##correlation using Kendal
Saudi_Arabia_kendall = Saudi_Arabia1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Saudi_Arabia_kendall, 'confirmed')
##correlation using Spearman
Saudi_Arabia_spearman = Saudi_Arabia1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Saudi_Arabia_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Saudi_Arabia2 = Saudi_Arabia.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Saudi_Arabia_pearson = Saudi_Arabia2.corr()
#heatmap_numeric_w_dependent_variable(Saudi_Arabia_pearson, 'deaths')
##correlation using Kendal
Saudi_Arabia_kendall = Saudi_Arabia2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Saudi_Arabia_kendall, 'deaths')
##correlation using Spearman
Saudi_Arabia_spearman = Saudi_Arabia2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Saudi_Arabia_spearman, 'deaths')
#plt.show()


#Make Correlation for Sweden using Pearson & Kendal & Spearman
#default correlation is Pearson
Sweden1 = Sweden.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Sweden_pearson = Sweden1.corr()
#heatmap_numeric_w_dependent_variable(Sweden_pearson, 'confirmed')
##correlation using Kendal
Sweden_kendall = Sweden1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Sweden_kendall, 'confirmed')
##correlation using Spearman
Sweden_spearman = Sweden1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Sweden_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Sweden2 = Sweden.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Sweden_pearson = Sweden2.corr()
#heatmap_numeric_w_dependent_variable(Sweden_pearson, 'deaths')
##correlation using Kendal
Sweden_kendall = Sweden2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Sweden_kendall, 'deaths')
##correlation using Spearman
Sweden_spearman = Sweden2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Sweden_spearman, 'deaths')
#plt.show()


#Make Correlation for US using Pearson & Kendal & Spearman
#default correlation is Pearson
US1 = US.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
US_pearson = US1.corr()
#heatmap_numeric_w_dependent_variable(US_pearson, 'confirmed')
##correlation using Kendal
US_kendall = US1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(US_kendall, 'confirmed')
##correlation using Spearman
US_spearman = US1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(US_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
US2 = US.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
US_pearson = US2.corr()
#heatmap_numeric_w_dependent_variable(US_pearson, 'deaths')
##correlation using Kendal
US_kendall = US2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(US_kendall, 'deaths')
##correlation using Spearman
US_spearman = US2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(US_spearman, 'deaths')
#plt.show()


#Make Correlation for UAE using Pearson & Kendal & Spearman
#default correlation is Pearson
UAE1 = UAE.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
UAE_pearson = UAE1.corr()
#heatmap_numeric_w_dependent_variable(UAE_pearson, 'confirmed')
##correlation using Kendal
UAE_kendall = UAE1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(UAE_kendall, 'confirmed')
##correlation using Spearman
UAE_spearman = UAE1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(UAE_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
UAE2 = UAE.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
UAE_pearson = UAE2.corr()
#heatmap_numeric_w_dependent_variable(UAE_pearson, 'deaths')
##correlation using Kendal
UAE_kendall = UAE2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(UAE_kendall, 'deaths')
##correlation using Spearman
UAE_spearman = UAE2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(UAE_spearman, 'deaths')
#plt.show()


#Make Correlation for China using Pearson & Kendal & Spearman
#default correlation is Pearson
China1 = China.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
China_pearson = China1.corr()
#heatmap_numeric_w_dependent_variable(China_pearson, 'confirmed')
##correlation using Kendal
China_kendall = China1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(China_kendall, 'confirmed')
##correlation using Spearman
China_spearman = China1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(China_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
China2 = China.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
China_pearson = China2.corr()
#heatmap_numeric_w_dependent_variable(China_pearson, 'deaths')
##correlation using Kendal
China_kendall = China2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(China_kendall, 'deaths')
##correlation using Spearman
China_spearman = China2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(China_spearman, 'deaths')
#plt.show()

#default correlation is Pearson
Norway1 = Norway.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Norway_pearson = Norway1.corr()
#heatmap_numeric_w_dependent_variable(Norway_pearson, 'confirmed')
##correlation using Kendal
Norway_kendall = Norway1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Norway_kendall, 'confirmed')
##correlation using Spearman
Norway_spearman = Norway1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Norway_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Norway2 = Norway.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Norway_pearson = Norway2.corr()
#heatmap_numeric_w_dependent_variable(Norway_pearson, 'deaths')
##correlation using Kendal
Norway_kendall = Norway2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Norway_kendall, 'deaths')
##correlation using Spearman
Norway_spearman = Norway2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Norway_spearman, 'deaths')
#plt.show()

#default correlation is Pearson
Iceland1 = Iceland.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Iceland_pearson = Iceland1.corr()
#heatmap_numeric_w_dependent_variable(Iceland_pearson, 'confirmed')
##correlation using Kendal
Iceland_kendall = Iceland1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Iceland_kendall, 'confirmed')
##correlation using Spearman
Iceland_spearman = Iceland1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Iceland_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Iceland2 = Iceland.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Iceland_pearson =Iceland2.corr()
#heatmap_numeric_w_dependent_variable(Iceland_pearson, 'deaths')
##correlation using Kendal
Iceland_kendall = Iceland2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Iceland_kendall, 'deaths')
##correlation using Spearman
Iceland_spearman = Iceland2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Iceland_spearman, 'deaths')
#plt.show()

#default correlation is Pearson
Finland1 = Finland.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Finland_pearson = Finland1.corr()
#heatmap_numeric_w_dependent_variable(Finland_pearson, 'confirmed')
##correlation using Kendal
Finland_kendall = Finland1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Finland_kendall, 'confirmed')
##correlation using Spearman
Finland_spearman = Finland1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Finland_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Finland2 = Finland.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Finland_pearson =Finland2.corr()
#heatmap_numeric_w_dependent_variable(Finland_pearson, 'deaths')
##correlation using Kendal
Finland_kendall = Finland2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Finland_kendall, 'deaths')
##correlation using Spearman
Finland_spearman = Finland2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Finland_spearman, 'deaths')
#plt.show()

#default correlation is Pearson
Canada1 = Canada.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Canada_pearson = Canada1.corr()
#heatmap_numeric_w_dependent_variable(Canada_pearson, 'confirmed')
##correlation using Kendal
Canada_kendall = Canada1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Canada_kendall, 'confirmed')
##correlation using Spearman
Canada_spearman = Canada1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Canada_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Canada2 = Canada.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Canada_pearson =Canada2.corr()
#heatmap_numeric_w_dependent_variable(Canada_pearson, 'deaths')
##correlation using Kendal
Canada_kendall = Canada2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Canada_kendall, 'deaths')
##correlation using Spearman
Canada_spearman = Canada2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Canada_spearman, 'deaths')
#plt.show()

#default correlation is Pearson
Brazil1 = Brazil.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Brazil_pearson = Brazil1.corr()
#heatmap_numeric_w_dependent_variable(Brazil_pearson, 'confirmed')
##correlation using Kendal
Brazil_kendall = Brazil1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Brazil_kendall, 'confirmed')
##correlation using Spearman
Brazil_spearman = Brazil1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Brazil_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Brazil2 = Brazil.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Brazil_pearson =Brazil2.corr()
#heatmap_numeric_w_dependent_variable(Brazil_pearson, 'deaths')
##correlation using Kendal
Brazil_kendall = Brazil2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Brazil_kendall, 'deaths')
##correlation using Spearman
Brazil_spearman = Brazil2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Brazil_spearman, 'deaths')
#plt.show()

#default correlation is Pearson
India1 = India.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
India_pearson = India1.corr()
#heatmap_numeric_w_dependent_variable(India_pearson, 'confirmed')
##correlation using Kendal
India_kendall = India1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(India_kendall, 'confirmed')
##correlation using Spearman
India_spearman = India1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(India_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
India2 = India.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
India_pearson =India2.corr()
#heatmap_numeric_w_dependent_variable(India_pearson, 'deaths')
##correlation using Kendal
India_kendall = India2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(India_kendall, 'deaths')
##correlation using Spearman
India_spearman = India2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(India_spearman, 'deaths')
#plt.show()


#default correlation is Pearson
Denmark1 = Denmark.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','deaths'], axis=1)
Denmark_pearson = Denmark1.corr()
#heatmap_numeric_w_dependent_variable(Denmark_pearson, 'confirmed')
##correlation using Kendal
Denmark_kendall = Denmark1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Denmark_kendall, 'confirmed')
##correlation using Spearman
Denmark_spearman = Denmark1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Denmark_spearman, 'confirmed')
#plt.show()

#default correlation is Pearson
Denmark2 = Denmark.drop(['Province/State', 'Country/Region','Lat','Long','covid_date','confirmed'], axis=1)
Denmark_pearson =Denmark2.corr()
#heatmap_numeric_w_dependent_variable(Denmark_pearson, 'deaths')
##correlation using Kendal
Denmark_kendall = Denmark2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Denmark_kendall, 'deaths')
##correlation using Spearman
Denmark_spearman = Denmark2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Denmark_spearman, 'deaths')
#plt.show()

# Correlation between attributes in COVID-19 pandemic
weather_corr = weather_df.drop(['Province/State', 'Country/Region','Lat','Long','covid_date'], axis=1)
#default correlation is Pearson
corr = weather_corr.corr()

fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
ax = sns.heatmap(corr,annot = True,cmap='Blues',square=True)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
ax.set_title('Correlation of different attributes in COVID-19 with Weather ')
plt.show()


#Get Correlations with Confirmed
heatmap_numeric_w_dependent_variable(corr, 'confirmed')
#Get Correlations with Deaths
#(corr, 'deaths')
plt.show()

