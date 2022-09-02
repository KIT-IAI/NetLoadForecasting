import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# generate Summertime
#

class DatamodifierSummertime(Datamodifier):
    def apply(self, df_old, params_dict=None):
        df_old = pd.DataFrame(df_old)

        df_old.loc[
            ((df_old.index >= '28-03-2010 02:00:00') & (df_old.index <= '31-10-2010 03:00:00')), 'summertime'] = 1
        df_old.loc[
            ((df_old.index >= '27-03-2011 02:00:00') & (df_old.index <= '30-10-2011 03:00:00')), 'summertime'] = 1
        df_old.loc[
            ((df_old.index >= '25-03-2012 02:00:00') & (df_old.index <= '28-10-2012 03:00:00')), 'summertime'] = 1
        df_old.loc[
            ((df_old.index >= '31-03-2013 02:00:00') & (df_old.index <= '27-10-2013 03:00:00')), 'summertime'] = 1
        df_old.loc[
            ((df_old.index >= '30-03-2014 02:00:00') & (df_old.index <= '26-10-2014 03:00:00')), 'summertime'] = 1
        df_old.loc[
            ((df_old.index >= '29-03-2015 02:00:00') & (df_old.index <= '25-10-2015 03:00:00')), 'summertime'] = 1
        df_old.loc[
            ((df_old.index >= '27-03-2016 02:00:00') & (df_old.index <= '30-10-2016 03:00:00')), 'summertime'] = 1
        df_old.loc[
            ((df_old.index >= '26-03-2017 02:00:00') & (df_old.index <= '29-10-2017 03:00:00')), 'summertime'] = 1
        df_old.loc[
            ((df_old.index >= '25-03-2018 02:00:00') & (df_old.index <= '28-10-2018 03:00:00')), 'summertime'] = 1
        df_old.loc[
            ((df_old.index >= '31-03-2019 02:00:00') & (df_old.index <= '27-10-2019 03:00:00')), 'summertime'] = 1

        cols_fillna = ['summertime']
        for col in cols_fillna:
            df_old[col].fillna(0, inplace=True)

        return df_old
