import pandas as pd


#
#
#
# Not modelled by datamodifiere, because of time-complexity joined on time key by weather-wind
#   it generates the weather features for solar ssr t2m str
#
#
#
#

#
# Mittelwert über alle werte in einem Bereich [lat_min,lat_max] ,[lon_min,lon_max]
#
def aggregate_between(data_to_modify, lat_min, lat_max, long_min, long_max):
    data = data_to_modify[(data_to_modify['latitude'] <= lat_max) & (data_to_modify['latitude'] >= lat_min) & (
                data_to_modify['longitude'] <= long_max) & (data_to_modify['longitude'] >= long_min)]
    data = data.groupby('time').mean()
    data = data.drop(['latitude', 'longitude'], axis=1)
    return data


if __name__ == '__main__':
    # df = pd.read_pickle("Germany_Reanalysis_Data_Solar_ONLY_2013-2014_with_temperature.pkl")
    list_args = ['latitude', 'longitude', 'time', 'ssr', 'str', 'ssrd', 'strd', 't2m']

    for x in range(2010, 2020):
        df = pd.read_csv("Weather_Germany_" + str(x) + ".csv", usecols=list_args)

        # print(data_german_weather.loc[(data_german_weather['step'] == "1 days 00:00:00") & (data_german_weather["latitude"] == 50) & (data_german_weather["longitude"] == 5)].head(50))

        # 51 - 54 und 6 - 14
        df_north = aggregate_between(df, 51, 54, 6, 14).rename(
            columns={"ssr": "ssr_north", "str": "str_north", "ssrd": "ssrd_north", "strd": "strd_north",
                     "t2m": "t2m_north"})

        # 50 - 51 und 6 - 14
        df_middle = aggregate_between(df, 50, 51, 6, 14).rename(
            columns={"ssr": "ssr_middle", "str": "str_middle", "ssrd": "ssrd_middle", "strd": "strd_middle",
                     "t2m": "t2m_middle"})

        # 47 - 50  und 7 - 13
        df_south = aggregate_between(df, 47, 50, 7, 13).rename(
            columns={"ssr": "ssr_south", "str": "str_south", "ssrd": "ssrd_south", "strd": "strd_south",
                     "t2m": "t2m_south"})

        df_combined = df_north.join(df_middle, on="time").join(df_south, on="time")

        df_combined.to_pickle("Germany_Reanalysis_Data_Solar_Aggregated_" + str(x) + ".pkl")
        df_combined.to_csv("Germany_Reanalysis_Data_Solar_Aggregated_" + str(x) + ".csv")
