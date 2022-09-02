import os
from glob import glob

import pandas as pd

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname)
    file_demand = glob(os.path.join(folder, 'demand/*.pkl'))[0]
    file_supply = glob(os.path.join(folder, 'supply/*.pkl'))[0]

    #
    # gt mismatch data
    #
    file_mismatch = os.path.join(folder, "mismatch/mismatch_0.csv")

    #
    # Fill in the results of the forecastin framework as folders for example demand with result.pkl inside
    #
    df_demand = pd.read_pickle(file_demand)[['demand', 'demand_hat']]
    df_supply = pd.read_pickle(file_supply)[["supply_re_hat"]]
    df_mismatch = pd.read_csv(file_mismatch, usecols=['name', 'mismatch_demand_renewable'],
                              parse_dates=[0])

    # here the produced data from forecasting framework for mismatch has to be
    df_combined = df_mismatch.join(df_demand, on="name").join(df_supply, on="name")
    df_combined["mismatch_combined_hat"] = df_combined['demand_hat'] - df_combined['supply_re_hat']
    df_combined = df_combined.set_index('name')[-8760:]

    df_combined.to_pickle(os.path.join(folder, "mismatch_combined_result.pkl"))
    df_combined.to_csv(os.path.join(folder, "mismatch_combined_result.csv"))
