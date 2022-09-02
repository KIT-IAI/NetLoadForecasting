from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# generate hours
#

class DatamodifierHours(Datamodifier):

    def apply(self, df_old, params_dict=None):
        df_old.loc[(df_old.index.hour == 0), 'hour-0'] = 1
        df_old.loc[(df_old.index.hour == 1), 'hour-1'] = 1
        df_old.loc[(df_old.index.hour == 2), 'hour-2'] = 1
        df_old.loc[(df_old.index.hour == 3), 'hour-3'] = 1
        df_old.loc[(df_old.index.hour == 4), 'hour-4'] = 1
        df_old.loc[(df_old.index.hour == 5), 'hour-5'] = 1
        df_old.loc[(df_old.index.hour == 6), 'hour-6'] = 1
        df_old.loc[(df_old.index.hour == 7), 'hour-7'] = 1
        df_old.loc[(df_old.index.hour == 8), 'hour-8'] = 1
        df_old.loc[(df_old.index.hour == 9), 'hour-9'] = 1
        df_old.loc[(df_old.index.hour == 10), 'hour-10'] = 1
        df_old.loc[(df_old.index.hour == 11), 'hour-11'] = 1
        df_old.loc[(df_old.index.hour == 12), 'hour-12'] = 1
        df_old.loc[(df_old.index.hour == 13), 'hour-13'] = 1
        df_old.loc[(df_old.index.hour == 14), 'hour-14'] = 1
        df_old.loc[(df_old.index.hour == 15), 'hour-15'] = 1
        df_old.loc[(df_old.index.hour == 16), 'hour-16'] = 1
        df_old.loc[(df_old.index.hour == 17), 'hour-17'] = 1
        df_old.loc[(df_old.index.hour == 18), 'hour-18'] = 1
        df_old.loc[(df_old.index.hour == 19), 'hour-19'] = 1
        df_old.loc[(df_old.index.hour == 20), 'hour-20'] = 1
        df_old.loc[(df_old.index.hour == 21), 'hour-21'] = 1
        df_old.loc[(df_old.index.hour == 22), 'hour-22'] = 1
        df_old.loc[(df_old.index.hour == 23), 'hour-23'] = 1

        cols_fillna = ['hour-0', 'hour-1', 'hour-2', 'hour-3', 'hour-4', 'hour-5', 'hour-6', 'hour-7', 'hour-8',
                       'hour-9', 'hour-10', 'hour-11', 'hour-12', 'hour-13', 'hour-14', 'hour-15', 'hour-16',
                       'hour-17', 'hour-18', 'hour-19', 'hour-20', 'hour-21', 'hour-22', 'hour-23']
        for col in cols_fillna:
            df_old[col].fillna(0, inplace=True)

        return df_old
