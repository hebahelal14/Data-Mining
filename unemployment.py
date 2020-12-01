import pandas as pd
import re as re
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

def heatmap_numeric_w_dependent_variable(df, dependent_variable):

    plt.figure(figsize=(8, 10))
    g = sns.heatmap(df[[dependent_variable]].sort_values(by=dependent_variable),
                    annot=True,
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1)
    print(df)
    return g

#Dataset of covid from main.py
covid = pd.read_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\covid_df+candinavian.csv')
#Convert date from string to numeric value which DM can deal with it
le = preprocessing.LabelEncoder()
covid['day_num'] = le.fit_transform(covid.covid_date)
covid['day_num'] = pd.to_datetime(covid['covid_date'], errors='coerce')
covid['day'] = covid['day_num'].dt.day
#print(covid['day_num'])
covid['month'] = covid['day_num'].dt.month
#print(covid['month'])
covid['year'] = covid['day_num'].dt.year

#get average of confirmed and death in each month
df = covid.groupby(['Country/Region',covid['month']], as_index=False)['confirmed'].mean()
df2 = covid.groupby(['Country/Region',covid['month']], as_index=False)['deaths'].mean()
covid_avg = pd.merge(df, df2,on=['Country/Region','month'])
covid_avg.rename (columns= { 'confirmed': 'confirmed_avg','deaths':'deaths_avg' }, inplace= True)
covid_avg.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\covid_avg.csv')


#Get Unemployment dataset from [Organization for Economic Co-Operation and Development][https://stats.oecd.org/index.aspx]
unemployment = pd.read_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\unemployment_df_add.csv')
unemployment.rename (columns= {'Value': 'Value %','Country':'Country/Region'}, inplace= True)

#Merge Covid with unemployment
covid_unemployment = pd.merge(covid_avg, unemployment ,on=['Country/Region','month'])
covid_unemployment.drop([ 'Unnamed: 0','Unnamed: 4','Unnamed: 5','Unnamed: 6'], axis=1,inplace=True)
#covid_unemployment.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\covid_unemployment_new.csv')

#Get Correlation with Confirmed
#default correlation is Pearson
#covid_unemployment = covid_unemployment.corr()
#print(covid_unemployment)
#heatmap_numeric_w_dependent_variable(covid_unemployment, 'confirmed_avg')
##correlation using Kendal
#covid_unemployment_kendall = covid_unemployment.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(covid_unemployment_kendall, 'confirmed_avg')
##correlation using Spearman
#covid_unemployment_spearman = covid_unemployment.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(covid_unemployment_spearman, 'confirmed_avg')
#plt.show()

#Get Correlation with Death
#default correlation is Pearson
#covid_unemployment = covid_unemployment.corr()
#heatmap_numeric_w_dependent_variable(covid_unemployment, 'deaths_avg')
##correlation using Kendal
#covid_unemployment_kendall = covid_unemployment.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(covid_unemployment_kendall, 'deaths_avg')
##correlation using Spearman
#covid_unemployment_spearman = covid_unemployment.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(covid_unemployment_spearman, 'deaths_avg')
#plt.show()

#default correlation is Pearson
#corr = covid_unemployment.corr()

#fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
#ax = sns.heatmap(corr,annot = True,cmap='Blues',square=True)

#ax.set_xticklabels(
#    ax.get_xticklabels(),
#    rotation=45,
#    horizontalalignment='right')
#ax.set_title('Correlation of different attributes in COVID-19 with Unemployment Rate ')
#plt.show()

#print(covid_unemployment['Country/Region'])
##select country Canada from total dataset
Canada = covid_unemployment.loc[covid_unemployment['Country/Region']=='Canada']
Canada.reset_index(drop=True, inplace=True)
#confirmed_avg = Canada['confirmed_avg'].values
#confirmed_avg=confirmed_avg.reshape(-1, 1)
#min_max_scaler = preprocessing.MinMaxScaler()
#confirmed_avg_scaled = min_max_scaler.fit_transform(confirmed_avg)
#Value = Canada['Value %'].values
#Value=Value.reshape(-1, 1)
#min_max_scaler = preprocessing.MinMaxScaler()
#Value_scaled = min_max_scaler.fit_transform(Value)
#Canada['confirmed_avg'] = confirmed_avg_scaled
#Canada['Value %'] = Value_scaled
#Canada.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Canada_nor.csv')
##select country Denmark from total dataset
Denmark = covid_unemployment.loc[covid_unemployment['Country/Region']=='Denmark']
Denmark.reset_index(drop=True, inplace=True)
#Denmark.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Denmark.csv')
##select country Finland from total dataset
Finland = covid_unemployment.loc[covid_unemployment['Country/Region']=='Finland']
Finland.reset_index(drop=True, inplace=True)
#Finland.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Finland.csv')
##select country Iceland from total dataset
Iceland = covid_unemployment.loc[covid_unemployment['Country/Region']=='Iceland']
Iceland.reset_index(drop=True, inplace=True)
#Iceland.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Iceland.csv')
##select country Norway from total dataset
Norway = covid_unemployment.loc[covid_unemployment['Country/Region']=='Norway']
Norway.reset_index(drop=True, inplace=True)
#Norway.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Norway.csv')
##select country Sweden from total dataset
Sweden = covid_unemployment.loc[covid_unemployment['Country/Region']=='Sweden']
Sweden.reset_index(drop=True, inplace=True)
#Sweden.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Sweden.csv')
##select country US from total dataset
US = covid_unemployment.loc[covid_unemployment['Country/Region']=='US']
US.reset_index(drop=True, inplace=True)
#US.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\US.csv')
##select country Bahrain from total dataset
Bahrain = covid_unemployment.loc[covid_unemployment['Country/Region']=='Bahrain']
Bahrain.reset_index(drop=True, inplace=True)
#Bahrain.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Bahrain.csv')
##select country Brazil from total dataset
Brazil = covid_unemployment.loc[covid_unemployment['Country/Region']=='Brazil']
Brazil.reset_index(drop=True, inplace=True)
#Brazil.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Brazil.csv')
##select country India from total dataset
India = covid_unemployment.loc[covid_unemployment['Country/Region']=='India']
India.reset_index(drop=True, inplace=True)
#India.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\India.csv')
##select country Kuwait from total dataset
Kuwait = covid_unemployment.loc[covid_unemployment['Country/Region']=='Kuwait']
Kuwait.reset_index(drop=True, inplace=True)
#Kuwait.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Kuwait.csv')
##select country Oman from total dataset
Oman = covid_unemployment.loc[covid_unemployment['Country/Region']=='Oman']
Oman.reset_index(drop=True, inplace=True)
#Oman.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Oman.csv')
##select country Saudi_Arabia from total dataset
Saudi_Arabia = covid_unemployment.loc[covid_unemployment['Country/Region']=='Saudi Arabia']
Saudi_Arabia.reset_index(drop=True, inplace=True)
#Saudi_Arabia.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Saudi_Arabia.csv')
##select country UAE from total dataset
UAE = covid_unemployment.loc[covid_unemployment['Country/Region']=='United Arab Emirates']
UAE.reset_index(drop=True, inplace=True)
#UAE.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\UAE.csv')
##select country Qatar from total dataset
Qatar = covid_unemployment.loc[covid_unemployment['Country/Region']=='Qatar']
Qatar.reset_index(drop=True, inplace=True)
#Qatar.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Unemployment-dataset\\Countries\\Qatar.csv')



#Make Correlation for Canada using Pearson & Kendal & Spearman
#default correlation is Pearson
Canada1 = Canada.drop(['Country/Region','month','deaths_avg'], axis=1)
#print(Canada1)
Canada_pearson = Canada1.corr()
#print(Canada_pearson)
#heatmap_numeric_w_dependent_variable(Canada_pearson,'confirmed_avg')
##correlation using Kendal
Canada_kendall = Canada1.corr(method = 'kendall')
#print(Canada_kendall)
#heatmap_numeric_w_dependent_variable(Canada_kendall, 'confirmed_avg')
##correlation using Spearman
Canada_spearman = Canada1.corr(method = 'spearman')
#print(Canada_spearman)
#heatmap_numeric_w_dependent_variable(Canada_spearman, 'confirmed_avg')
#plt.show()

#default correlation is Pearson
Canada2 = Canada.drop(['Country/Region','month','confirmed_avg'], axis=1)
Canada_pearson = Canada2.corr()
#print(Canada_pearson)
#heatmap_numeric_w_dependent_variable(Canada_pearson, 'deaths_avg')
##correlation using Kendal
Canada_kendall = Canada2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Canada_kendall, 'deaths_avg')
##correlation using Spearman
Canada_spearman = Canada2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Canada_spearman, 'deaths_avg')
#plt.show()

#Make Correlation for Denmark using Pearson & Kendal & Spearman
#default correlation is Pearson
Denmark1 = Denmark.drop(['Country/Region','month','deaths_avg'], axis=1)
Denmark_pearson = Denmark1.corr()
#heatmap_numeric_w_dependent_variable(Denmark_pearson, 'confirmed_avg')
##correlation using Kendal
Denmark_kendall = Denmark1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Denmark_kendall, 'confirmed_avg')
##correlation using Spearman
Denmark_spearman = Denmark1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Denmark_spearman, 'confirmed_avg')
#plt.show()

#default correlation is Pearson
Denmark2 = Denmark.drop(['Country/Region','month','confirmed_avg'], axis=1)
Denmark_pearson = Denmark2.corr()
#heatmap_numeric_w_dependent_variable(Denmark_pearson, 'deaths_avg')
##correlation using Kendal
Denmark_kendall = Denmark2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Denmark_kendall, 'deaths_avg')
##correlation using Spearman
Denmark_spearman = Denmark2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Denmark_spearman, 'deaths_avg')
#plt.show()


#Make Correlation for Finland using Pearson & Kendal & Spearman
#default correlation is Pearson
Finland1 = Finland.drop(['Country/Region','month','deaths_avg'], axis=1)
Finland_pearson = Finland1.corr()
#heatmap_numeric_w_dependent_variable(Finland_pearson, 'confirmed_avg')
##correlation using Kendal
Finland_kendall = Finland1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Finland_kendall, 'confirmed_avg')
##correlation using Spearman
Finland_spearman = Finland1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Finland_spearman, 'confirmed_avg')
#plt.show()

#default correlation is Pearson
Finland2 = Finland.drop(['Country/Region','month','confirmed_avg'], axis=1)
Finland_pearson = Finland2.corr()
#heatmap_numeric_w_dependent_variable(Finland_pearson, 'deaths_avg')
##correlation using Kendal
Finland_kendall = Finland2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Finland_kendall, 'deaths_avg')
##correlation using Spearman
Finland_spearman = Finland2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Finland_spearman, 'deaths_avg')
#plt.show()


#Make Correlation for Iceland using Pearson & Kendal & Spearman
#default correlation is Pearson
Iceland1 = Iceland.drop(['Country/Region','month','deaths_avg'], axis=1)
Iceland_pearson = Iceland1.corr()
#heatmap_numeric_w_dependent_variable(Iceland_pearson, 'confirmed_avg')
##correlation using Kendal
Iceland_kendall = Iceland1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Iceland_kendall, 'confirmed_avg')
##correlation using Spearman
Iceland_spearman = Iceland1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Iceland_spearman, 'confirmed_avg')
#plt.show()

#default correlation is Pearson
Iceland2 = Iceland.drop(['Country/Region','month','confirmed_avg'], axis=1)
Iceland_pearson = Iceland2.corr()
#heatmap_numeric_w_dependent_variable(Iceland_pearson, 'deaths_avg')
##correlation using Kendal
Iceland_kendall = Iceland2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Iceland_kendall, 'deaths_avg')
##correlation using Spearman
Iceland_spearman = Iceland2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Iceland_spearman, 'deaths_avg')
#plt.show()


#Make Correlation for Norway using Pearson & Kendal & Spearman
#default correlation is Pearson
Norway1 = Norway.drop(['Country/Region','month','deaths_avg'], axis=1)
Norway_pearson = Norway1.corr()
#heatmap_numeric_w_dependent_variable(Norway_pearson, 'confirmed_avg')
##correlation using Kendal
Norway_kendall = Norway1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Norway_kendall, 'confirmed_avg')
##correlation using Spearman
Norway_spearman = Norway1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Norway_spearman, 'confirmed_avg')
#plt.show()

#default correlation is Pearson
Norway2 = Norway.drop(['Country/Region','month','confirmed_avg'], axis=1)
Norway_pearson = Norway2.corr()
#heatmap_numeric_w_dependent_variable(Norway_pearson, 'deaths_avg')
##correlation using Kendal
Norway_kendall = Norway2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Norway_kendall, 'deaths_avg')
##correlation using Spearman
Norway_spearman = Norway2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Norway_spearman, 'deaths_avg')
#plt.show()


#Make Correlation for Sweden using Pearson & Kendal & Spearman
#default correlation is Pearson
Sweden1 = Sweden.drop(['Country/Region','month','deaths_avg'], axis=1)
Sweden_pearson = Sweden1.corr()
#heatmap_numeric_w_dependent_variable(Sweden_pearson, 'confirmed_avg')
##correlation using Kendal
Sweden_kendall = Sweden1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Sweden_kendall, 'confirmed_avg')
##correlation using Spearman
Sweden_spearman = Sweden1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Sweden_spearman, 'confirmed_avg')
#plt.show()

#default correlation is Pearson
Sweden2 = Sweden.drop(['Country/Region','month','confirmed_avg'], axis=1)
Sweden_pearson = Sweden2.corr()
#heatmap_numeric_w_dependent_variable(Sweden_pearson, 'deaths_avg')
##correlation using Kendal
Sweden_kendall = Sweden2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(Sweden_kendall, 'deaths_avg')
##correlation using Spearman
Sweden_spearman = Sweden2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(Sweden_spearman, 'deaths_avg')
#plt.show()


#Make Correlation for US using Pearson & Kendal & Spearman
#default correlation is Pearson
US1 = US.drop(['Country/Region','month','deaths_avg'], axis=1)
US_pearson = US1.corr()
#heatmap_numeric_w_dependent_variable(US_pearson, 'confirmed_avg')
##correlation using Kendal
US_kendall = US1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(US_kendall, 'confirmed_avg')
##correlation using Spearman
US_spearman = US1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(US_spearman, 'confirmed_avg')
#plt.show()

#default correlation is Pearson
US2 = US.drop(['Country/Region','month','confirmed_avg'], axis=1)
US_pearson = US2.corr()
#heatmap_numeric_w_dependent_variable(US_pearson, 'deaths_avg')
##correlation using Kendal
US_kendall = US2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(US_kendall, 'deaths_avg')
##correlation using Spearman
US_spearman = US2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(US_spearman, 'deaths_avg')
#plt.show()

#Make Correlation for Bahrain using Pearson & Kendal & Spearman
#default correlation is Pearson
#Bahrain1 = Bahrain.drop(['Country/Region','month','deaths_avg'], axis=1)
#Bahrain_pearson = Bahrain1.corr()
#heatmap_numeric_w_dependent_variable(Bahrain_pearson, 'confirmed_avg')
##correlation using Kendal
#US_kendall = US1.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(US_kendall, 'confirmed_avg')
##correlation using Spearman
#US_spearman = US1.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(US_spearman, 'confirmed_avg')
#plt.show()

#default correlation is Pearson
#Bahrain2 = Bahrain.drop(['Country/Region','month','confirmed_avg'], axis=1)
#Bahrain_pearson = Bahrain2.corr()
#heatmap_numeric_w_dependent_variable(Bahrain_pearson, 'deaths_avg')
##correlation using Kendal
#US_kendall = US2.corr(method = 'kendall')
#heatmap_numeric_w_dependent_variable(US_kendall, 'deaths_avg')
##correlation using Spearman
#US_spearman = US2.corr(method = 'spearman')
#heatmap_numeric_w_dependent_variable(US_spearman, 'deaths_avg')
#plt.show()