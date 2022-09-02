import os

import pandas as pd
from forecasting_framework.datamodifier.datamodifier import Datamodifier


#
# Generate public holdiay feature and partial public holiday from /datamodifier/public_holiday_2010-2018.csv
#
class DatamodifierHoliday(Datamodifier):

    def apply(self, df_old, params_dict=None):
        df_old = pd.DataFrame(df_old)

        parser = lambda date: pd.datetime.strptime(date, '%d.%m.%Y')

        df_public_holiday_2010_2013 = pd.read_csv(os.getcwd() + "/datamodifier/public_holiday_2010-2019.csv",
                                                  parse_dates=[0], date_parser=parser,
                                                  usecols=["date", "public_Holiday", "gesetzl"],
                                                  index_col="date")

        for index, row in df_public_holiday_2010_2013.iterrows():
            df_old.loc[(df_old.index.date == index), 'public_holiday'] = row['public_Holiday']
            if row['public_Holiday'] == 0 and row['gesetzl'] == 1:
                df_old.loc[(df_old.index.date == index), 'partial_public_holiday'] = 1

        cols_fillna = ['public_holiday', 'partial_public_holiday']
        for col in cols_fillna:
            df_old[col].fillna(0, inplace=True)

        return df_old
