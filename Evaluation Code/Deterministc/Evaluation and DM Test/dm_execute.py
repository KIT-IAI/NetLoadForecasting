import pandas as pd

import dm_test

if __name__ == '__main__':
    #
    # Fill in the paths of every resulting pkl for mismatch
    #

    df_fully_disaggregated = pd.read_pickle("Path.pkl")
    df_partial_aggregated = pd.read_pickle("Path.pkl")
    df_fully_aggregated = pd.read_pickle("Path.pkl")

    print("Combined vs Supp_Demand", dm_test.dm_test(df_fully_aggregated["mismatch_demand_renewable"],
                                                     df_fully_disaggregated["mismatch_combined_hat"],
                                                     df_partial_aggregated["mismatch_combined_hat"]))
