import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# Calculate mismatch
#

class DatamodifierMismatchDemandRenewable(Datamodifier):
    def apply(self, df_old, params_dict=None):
        df_old = pd.DataFrame(df_old)
        df_old['mismatch_demand_renewable'] = df_old['demand'] - df_old['supply_re']
        return df_old
