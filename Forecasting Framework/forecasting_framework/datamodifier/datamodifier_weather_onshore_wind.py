import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# Weather offshore new 2010-2020
#

class DatamodifierWeatherOnshoreWind2010_2020(Datamodifier):
    def apply(self, df_old, params_dict=None):
        df_old = pd.DataFrame(df_old)
        dfs = [
            pd.read_csv("datamodifier/Germany_Reanalysis_Data_Wind_Aggregated_Onshore_" + str(x) + ".csv", index_col=0)
            for x in range(2010, 2020)]
        df_old = df_old.join(pd.concat(dfs))
        return df_old
