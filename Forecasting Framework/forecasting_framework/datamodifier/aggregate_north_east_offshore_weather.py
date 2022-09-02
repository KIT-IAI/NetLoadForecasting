import pandas as pd


#
#
#
# Not modelled by datamodifiere, because of time-complexity joined on time key by weather-wind
#   it generates the weather features for wind offshore
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
    return data


if __name__ == '__main__':
    list_args = ["latitude", "longitude", "time", "u100", "v100"]
    for x in range(2010, 2020):
        df = pd.read_csv("Weather_Germany_" + str(x) + ".csv", usecols=list_args)

        # print(data_german_weather.loc[(data_german_weather['step'] == "1 days 00:00:00") & (data_german_weather["latitude"] == 50) & (data_german_weather["longitude"] == 5)].head(50))

        #
        # Hier werden die Rechtecke gesteckt z.B Fall Nord 53,5 bis 55 und 4 bis 9
        #
        #
        df_north = aggregate_between(df, 53.5, 55, 4, 9).rename(columns={"u100": "u100_north", "v100": "v100_north"})
        df_east = aggregate_between(df, 53.5, 55, 9.5, 14).rename(columns={"u100": "u100_east", "v100": "v100_east"})

        df_north['length_north'] = (df_north['u100_north'] ** 2 + df_north['v100_north'] ** 2) ** (1 / 2)
        df_north['length_north_cubic'] = df_north['length_north'] ** 3

        df_east['length_east'] = (df_east['u100_east'] ** 2 + df_east['v100_east'] ** 2) ** (1 / 2)
        df_east['length_east_cubic'] = df_east['length_east'] ** 3

        df_east.loc[(df_east['length_east'] > 13), 'critical_strength_east'] = 1
        df_north.loc[(df_north['length_north'] > 13), 'critical_strength_north'] = 1

        # print(df_nord)

        df_north['critical_strength_north'].fillna(0, inplace=True)
        df_east['critical_strength_east'].fillna(0, inplace=True)

        df_north['critical_strength_north_scaled'] = (df_north['length_north'] / 9) ** 4
        df_east['critical_strength_east_scaled'] = (df_east['length_east'] / 9) ** 4

        df_combined = df_north[
            ["u100_north", "v100_north", "length_north", 'critical_strength_north', 'length_north_cubic',
             'critical_strength_north_scaled']].join(df_east[["u100_east", "v100_east", 'length_east',
                                                              'critical_strength_east', 'length_east_cubic',
                                                              'critical_strength_east_scaled']], on="time")

        df_combined.to_pickle("Germany_Reanalysis_Data_Wind_Aggregated_Offshore_" + str(x) + ".pkl")
        df_combined.to_csv("Germany_Reanalysis_Data_Wind_Aggregated_Offshore_" + str(x) + ".csv")
