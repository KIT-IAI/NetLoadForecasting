import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# supply re
#

class DatamodifierSupplyAggRenewable(Datamodifier):
    def apply(self, df_old, params_dict=None):
        df_old = pd.DataFrame(df_old)

        df_old['supply_re'] = df_old["ror"] + df_old["offwind-ac"] + df_old["offwind-dc"] + df_old["onwind"] + df_old[
            "solar"] + df_old["curtailment-offwind-ac"] + df_old["curtailment-offwind-dc"] + df_old[
                                  "curtailment-onwind"] + df_old["curtailment-ror"] + df_old["curtailment-solar"]

        return df_old
