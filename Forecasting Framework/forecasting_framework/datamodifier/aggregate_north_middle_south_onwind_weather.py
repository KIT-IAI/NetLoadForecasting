import pandas as pd


#
#
#
# Not modelled by datamodifiere, because of time-complexity joined on time key by weather-wind
#   it generates the weather features for wind onshore
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

    for x in range(2010, 2020):
        df = pd.read_csv("Weather_Germany_" + str(x) + ".csv",
                         usecols=["latitude", "longitude", "time", "u100", "v100"])
        print(df)

        # 51 - 54 und 6 - 14
        df_north = aggregate_between(df, 51, 54, 6, 14).rename(
            columns={"u100": "onwind_u100_north", "v100": "onwind_v100_north"})

        # 50 - 51 und 6 - 14
        df_middle = aggregate_between(df, 50, 51, 6, 14).rename(
            columns={"u100": "onwind_u100_middle", "v100": "onwind_v100_middle"})

        # 47 - 50  und 7 - 13
        df_south = aggregate_between(df, 47, 50, 7, 13).rename(
            columns={"u100": "onwind_u100_south", "v100": "onwind_v100_south"})

        df_north['onwind_length_north'] = (df_north['onwind_u100_north'] ** 2 + df_north['onwind_v100_north'] ** 2) ** (
                1 / 2)
        df_north['onwind_length_north_cubic'] = df_north['onwind_length_north'] ** 3

        df_middle['onwind_length_middle'] = (df_middle['onwind_u100_middle'] ** 2 + df_middle[
            'onwind_v100_middle'] ** 2) ** (1 / 2)
        df_middle['onwind_length_middle_cubic'] = df_middle['onwind_length_middle'] ** 3

        df_south['onwind_length_south'] = (df_south['onwind_u100_south'] ** 2 + df_south['onwind_v100_south'] ** 2) ** (
                1 / 2)
        df_south['onwind_length_south_cubic'] = df_south['onwind_length_south'] ** 3

        df_south.loc[(df_south['onwind_length_south'] > 13), 'onwind_critical_strength_south'] = 1
        df_middle.loc[(df_middle['onwind_length_middle'] > 13), 'onwind_critical_strength_middle'] = 1
        df_north.loc[(df_north['onwind_length_north'] > 13), 'onwind_critical_strength_north'] = 1

        # print(df_nord)

        df_north['onwind_critical_strength_north'].fillna(0, inplace=True)
        df_middle['onwind_critical_strength_middle'].fillna(0, inplace=True)
        df_south['onwind_critical_strength_south'].fillna(0, inplace=True)

        df_north['onwind_critical_strength_north_scaled'] = (df_north['onwind_length_north'] / 9) ** 4
        df_middle['onwind_critical_strength_middle_scaled'] = (df_middle['onwind_length_middle'] / 9) ** 4
        df_south['onwind_critical_strength_south_scaled'] = (df_south['onwind_length_south'] / 9) ** 4

        # print(df_ost)
        # print(df_nord)

        df_combined = df_north.join(df_middle, on="time").join(df_south, on="time")

        df_combined.to_csv("Germany_Reanalysis_Data_Wind_Aggregated_Onshore_" + str(x) + ".csv")
        df_combined.to_pickle("Germany_Reanalysis_Data_Wind_Aggregated_Onshore_" + str(x) + ".pkl")
