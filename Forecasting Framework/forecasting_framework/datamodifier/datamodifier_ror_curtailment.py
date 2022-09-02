import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# generate ror curtailment
#

class DatamodifierRoRCurtailment(Datamodifier):
    def apply(self, df_old, params_dict=None):
        df_old = pd.DataFrame(df_old)

        df_old['ror_with_curtailment'] = df_old['ror'] + df_old['curtailment-ror']

        return df_old
