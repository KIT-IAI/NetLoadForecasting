import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# solar curtailment
#
class DatamodifierSolarAgg(Datamodifier):
    def apply(self, df_old, params_dict=None):
        df_old = pd.DataFrame(df_old)

        # print(df.keys())

        df_old['solar_with_curtailment'] = df_old['solar'] + df_old['curtailment-solar']
        return df_old
