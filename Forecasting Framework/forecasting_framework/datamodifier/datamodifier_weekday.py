import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# generate weekday
#

class DatamodifierWeekday(Datamodifier):

    def apply(self, df_old, params_dict=None):
        df_old = pd.DataFrame(df_old)

        df_old.loc[(df_old.index.dayofweek == 0), 'monday'] = 1
        df_old.loc[(df_old.index.dayofweek == 1), 'tuesday'] = 1
        df_old.loc[(df_old.index.dayofweek == 2), 'wednesday'] = 1
        df_old.loc[(df_old.index.dayofweek == 3), 'thursday'] = 1
        df_old.loc[(df_old.index.dayofweek == 4), 'friday'] = 1
        df_old.loc[(df_old.index.dayofweek == 5), 'saturday'] = 1
        df_old.loc[(df_old.index.dayofweek == 6), 'sunday'] = 1

        df_old.loc[(df_old.index.dayofweek == 0), 'monday'] = 1
        df_old.loc[(df_old.index.dayofweek == 1), 'tuesday'] = 1
        df_old.loc[(df_old.index.dayofweek == 2), 'wednesday'] = 1
        df_old.loc[(df_old.index.dayofweek == 3), 'thursday'] = 1
        df_old.loc[(df_old.index.dayofweek == 4), 'friday'] = 1
        df_old.loc[(df_old.index.dayofweek == 5), 'saturday'] = 1
        df_old.loc[(df_old.index.dayofweek == 6), 'sunday'] = 1

        cols_fillna = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for col in cols_fillna:
            df_old[col].fillna(0, inplace=True)

        return df_old
