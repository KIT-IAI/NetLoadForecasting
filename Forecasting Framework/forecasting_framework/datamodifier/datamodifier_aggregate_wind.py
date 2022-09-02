import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# aggregate wind (old ) and generate onwind with curtailment
#

class DatamodifierWindAgg(Datamodifier):
    def apply(self, df_old, params_dict=None):
        df_old = pd.DataFrame(df_old)

        df_old['wind_agg'] = df_old['offwind-ac'] + df_old['offwind-dc'] + df_old['onwind']
        df_old['onwind_with_curtailment'] = df_old['onwind'] + df_old['curtailment-onwind']

        return df_old
