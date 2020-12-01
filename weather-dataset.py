from google.cloud import bigquery
import numpy as np
import pandas as pd
from scipy.spatial import distance



client = bigquery.Client()
dataset_ref = client.dataset("noaa_gsod", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)

tables = list(client.list_tables(dataset))

table_ref = dataset_ref.table("stations")
table = client.get_table(table_ref)
stations_df = client.list_rows(table).to_dataframe()

table_ref = dataset_ref.table("gsod2020")
table = client.get_table(table_ref)
twenty_twenty_df = client.list_rows(table).to_dataframe()

stations_df['STN'] = stations_df['usaf'] + '-' + stations_df['wban']
twenty_twenty_df['STN'] = twenty_twenty_df['stn'] + '-' + twenty_twenty_df['wban']

cols_1 = ['STN', 'mo', 'da', 'temp', 'min', 'max', 'stp', 'slp', 'dewp', 'wdsp', 'prcp', 'fog']
cols_2 = ['STN', 'country', 'state', 'call', 'lat', 'lon', 'elev']
weather_df = twenty_twenty_df[cols_1].join(stations_df[cols_2].set_index('STN'), on='STN')

#print(weather_df.head())
weather_df['temp'] = weather_df['temp'].apply(lambda x: np.nan if x==9999.9 else x)
weather_df['max'] = weather_df['max'].apply(lambda x: np.nan if x==9999.9 else x)
weather_df['min'] = weather_df['min'].apply(lambda x: np.nan if x==9999.9 else x)
weather_df['stp'] = weather_df['stp'].apply(lambda x: np.nan if x==9999.9 else x)
weather_df['slp'] = weather_df['slp'].apply(lambda x: np.nan if x==9999.9 else x)
weather_df['dewp'] = weather_df['dewp'].apply(lambda x: np.nan if x==9999.9 else x)
weather_df['wdsp'] = weather_df['wdsp'].apply(lambda x: np.nan if x==999.9 else x)
weather_df['prcp'] = weather_df['prcp'].apply(lambda x: np.nan if x==99.9 else x)

print(weather_df.tail(10))
weather_df.info(verbose=True)

# convert everything into celsius
temp = (weather_df['temp'] - 32) / 1.8
dewp = (weather_df['dewp'] - 32) / 1.8

# compute relative humidity as ratio between actual vapour pressure (computed from dewpoint temperature)
# and saturation vapour pressure (computed from temperature) (the constant 6.1121 cancels out)
weather_df['rh'] = (np.exp((18.678 * dewp) / (257.14 + dewp)) / np.exp((18.678 * temp) / (257.14 + temp)))

# calculate actual vapour pressure (in pascals)
# then use it to compute absolute humidity from the gas law of vapour
# (ah = mass / volume = pressure / (constant * temperature))
weather_df['ah'] = ((np.exp((18.678 * dewp) / (257.14 + dewp))) * 6.1121 * 100) / (461.5 * temp)

train = pd.read_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\covid_df.csv')  # Be Attention Id in covid_df dataset

mo = train['covid_date'].apply(lambda x: x[5:7])
da = train['covid_date'].apply(lambda x: x[8:10])

C = []
for j in train.index:
    df = train.iloc[j:(j+1)]
    mat = distance.cdist(df[['Lat','Long']],weather_df[['lat','lon']],metric='euclidean')
    new_df = pd.DataFrame(mat, index=df.Id, columns=weather_df.index)
    arr = new_df.values
    new_close = np.where(arr == np.nanmin(arr, axis=1)[:, None], new_df.columns, False)
    L = [i[i.astype(bool)].tolist()[0] for i in new_close]
    C.append(L[0])
train['closest_station'] = C

train = train.set_index('closest_station').join(
    weather_df[['temp', 'min', 'max', 'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(
    ['index'], axis=1)
train.sort_values(by=['Id'], inplace=True)
train.index = train['Id'].apply(lambda x: x - 1)
print(train.head())
#train.to_csv('E:\\PHD\\Courses\\Data-Mining\\Project\\datasets\\Clustering-new\\Weather+Covid-19\\weather_dataset3.csv')