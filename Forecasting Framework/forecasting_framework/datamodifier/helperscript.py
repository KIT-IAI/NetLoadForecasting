import pandas as pd

# script um 2013-2014 zu teilen
if __name__ == '__main__':
    df = pd.read_csv("Germany_Reanalysis_Data_2013-2014.csv")
    df_2014 = df[(df["time"] >= "2014-01-01")]
    # print(df_2013)
    df_2014.to_csv("Weather_Germany_2014.csv", index=False)
