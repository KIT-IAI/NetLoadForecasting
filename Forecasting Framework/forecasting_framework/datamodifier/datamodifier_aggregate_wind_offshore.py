import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# gennerate offshore wind
#
class DatamodifierWindAggOffshore(Datamodifier):
    def apply(self, df_old, params_dict=None):
        df_old = pd.DataFrame(df_old)

        # print(df.keys())

        df_old['wind_agg_off'] = df_old['offwind-ac'] + df_old['offwind-dc']

        df_old['wind_agg_off_with_curtailment'] = df_old['wind_agg_off'] + df_old['curtailment-offwind-ac'] + df_old[
            'curtailment-offwind-dc']

        return df_old
