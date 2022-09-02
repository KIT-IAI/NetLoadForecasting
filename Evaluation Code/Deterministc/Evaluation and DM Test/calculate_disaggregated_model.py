import os
from glob import glob

import pandas as pd

if __name__ == '__main__':
    #
    # Fill in the results of the forecasting framework as folders for example demand with result.pkl inside
    #
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname)

    file_demand = glob(os.path.join(folder, 'demand/*.pkl'))[0]
    file_offshore = glob(os.path.join(folder, 'offshore_agg_curtailment/*.pkl'))[0]
    file_onwind = glob(os.path.join(folder, 'onwind_curtailment/*.pkl'))[0]
    file_ror = glob(os.path.join(folder, 'ror_curtailment/*.pkl'))[0]
    file_solar = glob(os.path.join(folder, 'solar_curtailment/*.pkl'))[0]

    # here the produced data from forecasting framework for mismatch has to be
    file_mismatch = os.path.join(folder, "mismatch/mismatch_0.csv")

    df_demand = pd.read_pickle(file_demand)[['demand', 'demand_hat']]
    df_offshore = pd.read_pickle(file_offshore)[
        ['wind_agg_off_with_curtailment', 'wind_agg_off_with_curtailment_hat']]
    df_onwind = pd.read_pickle(file_onwind)[
        ['onwind_with_curtailment', 'onwind_with_curtailment_hat']]
    df_ror = pd.read_pickle(file_ror)[
        ["ror_with_curtailment", "ror_with_curtailment_hat"]]
    df_solar = pd.read_pickle(file_solar)[
        ["solar_with_curtailment", "solar_with_curtailment_hat"]]
    df_mismatch = pd.read_csv(file_mismatch, usecols=['name', 'mismatch_demand_renewable'],
                              parse_dates=[0])

    df_combined = df_mismatch.join(df_demand, on="name").join(df_offshore, on="name").join(df_onwind, on="name").join(
        df_ror, on="name").join(df_solar, on="name")
    df_combined["mismatch_combined_hat"] = df_combined['demand_hat'] - df_combined[
        'onwind_with_curtailment_hat'] - df_combined['wind_agg_off_with_curtailment_hat'] - df_combined[
                                               "ror_with_curtailment_hat"] - \
                                           df_combined["solar_with_curtailment_hat"]

    df_combined = df_combined.set_index('name')[-8760:]
    df_combined.to_pickle(os.path.join(folder, "mismatch_result.pkl"))
    df_combined.to_csv(os.path.join(folder, "mismatch_result.csv"))
