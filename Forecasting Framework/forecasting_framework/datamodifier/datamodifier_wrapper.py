import json
import os.path
from glob import glob

import pandas as pd
from forecasting_framework.datamodifier.datamodifier_aggregate_solar import DatamodifierSolarAgg
from forecasting_framework.datamodifier.datamodifier_aggregate_supply_renewable import DatamodifierSupplyAggRenewable
from forecasting_framework.datamodifier.datamodifier_aggregate_wind import DatamodifierWindAgg
from forecasting_framework.datamodifier.datamodifier_aggregate_wind_offshore import DatamodifierWindAggOffshore
from forecasting_framework.datamodifier.datamodifier_holiday import DatamodifierHoliday
from forecasting_framework.datamodifier.datamodifier_hours import DatamodifierHours
from forecasting_framework.datamodifier.datamodifier_lag_features import DatamodifierLag_Features
from forecasting_framework.datamodifier.datamodifier_mismatch_demand_renewable import \
    DatamodifierMismatchDemandRenewable
from forecasting_framework.datamodifier.datamodifier_ror_curtailment import DatamodifierRoRCurtailment
from forecasting_framework.datamodifier.datamodifier_summertime import DatamodifierSummertime
from forecasting_framework.datamodifier.datamodifier_weather_offshore_wind import \
    DatamodifierWeatherOffshoreWind_2010_2020
from forecasting_framework.datamodifier.datamodifier_weather_onshore_wind import DatamodifierWeatherOnshoreWind2010_2020
from forecasting_framework.datamodifier.datamodifier_weather_solar import DatamodifierWeatherSolar2010_2020
from forecasting_framework.datamodifier.datamodifier_weekday import DatamodifierWeekday

_folder_datamodifiers = '../datamodifiers'


#
# Sucht sich die jeweiligen Modifikatoren als json
#
def launch():
    _dirname = os.path.dirname(__file__)
    _folder = os.path.join(_dirname, _folder_datamodifiers)
    _pattern = os.path.join(_folder, '*.json')

    _folder_data = os.path.join(_folder, "data")

    for file_name in glob(_pattern):
        with open(file_name) as f:
            print(file_name)
            json_object = json.load(f)

            param_dict = json_object["param_dict"]
            modified = []

            if (json_object['csv']):
                modified = [pd.read_csv(os.path.join(_folder_data, json_object['datapath']), parse_dates=[0],
                                        usecols=json_object['cols'], index_col=json_object['indexcol'],
                                        float_precision='%.3f')]
                modified = modifiers(json_object, modified, param_dict)

            elif (json_object['pkl']):
                modified = [pd.read_pickle(os.path.join(_folder_data, json_object['datapath']))]
                modified = modifiers(json_object, modified, param_dict)

            if json_object['save_csv']:
                save_csv(modified, json_object['name'], os.path.join(_folder_data, json_object['name']))
            if json_object['save_pkl']:
                save_pkl(modified, json_object['name'], os.path.join(_folder_data, json_object['name']))


#
# Führt den jeweiligen Modifier aus
#

def modifiers(json_object, modified: [], param_dict=None):
    return_dfs = []
    for x in json_object['modifiers']:

        for i in range(len(modified)):
            # only data modifiers
            if x == "holiday":
                modified[i] = DatamodifierHoliday().apply(modified[i])
            elif x == "weekday":
                modified[i] = DatamodifierWeekday().apply(modified[i])
            elif x == "summertime":
                modified[i] = DatamodifierSummertime().apply(modified[i])
            elif x == 'wind_agg':
                modified[i] = DatamodifierWindAgg().apply(modified[i])
            elif x == 'wind_agg_offshore':
                modified[i] = DatamodifierWindAggOffshore().apply(modified[i])
            elif x == 'solar_agg':
                modified[i] = DatamodifierSolarAgg().apply(modified[i])
            elif x == 'supply_agg_renewable':
                modified[i] = DatamodifierSupplyAggRenewable().apply(modified[i])
            elif x == "mismatch_demand_renewable":
                modified[i] = DatamodifierMismatchDemandRenewable().apply(modified[i])
            elif x == "solar_weather":
                modified[i] = DatamodifierWeatherSolar2010_2020().apply(modified[i])
            elif x == "wind_weather_Offshore":
                modified[i] = DatamodifierWeatherOffshoreWind_2010_2020().apply(modified[i])
            elif x == "wind_weather_Onshore":
                modified[i] = DatamodifierWeatherOnshoreWind2010_2020().apply(modified[i])
            elif x == "ror_curtailment":
                modified[i] = DatamodifierRoRCurtailment().apply(modified[i])
            elif x == "hours":
                modified[i] = DatamodifierHours().apply(modified[i])
            elif x == "lag":
                modified[i] = DatamodifierLag_Features().apply(modified[i], params_dict=param_dict)



    return_dfs = modified
    return return_dfs


def save_pkl(dfs, name, folder_results):
    if not os.path.exists(folder_results):
        os.mkdir(folder_results)
    counter = 0
    for subdf in dfs:
        subdf.to_pickle(folder_results + "/" + name + "_" + str(counter) + ".pkl")
        counter = counter + 1


def save_csv(dfs, name, folder_results):
    if not os.path.exists(folder_results):
        os.mkdir(folder_results)
    counter = 0
    for subdf in dfs:
        subdf.to_csv(folder_results + "/" + name + "_" + str(counter) + ".csv")
        counter = counter + 1
