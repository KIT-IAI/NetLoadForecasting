import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# generate lag
#

class DatamodifierLag_Features(Datamodifier):

    def apply(self, df_old, params_dict=None):
        list_lag_features = []
        series = []

        forecast_variable = params_dict["forecasting_variable"]
        lag_count = 7
        for z in range(0, lag_count + 1):
            series.append(df_old[forecast_variable].shift((z + 1) * 24))
            list_lag_features.append("lag_" + str(z))

        lag_features = pd.concat(series, axis=1, join='inner')
        lag_features.columns = list_lag_features

        return pd.concat([df_old, lag_features], axis=1, join='inner')
