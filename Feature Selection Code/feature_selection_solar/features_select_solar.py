import json
import warnings

import pandas as pd
from matplotlib import pyplot as plt
from numpy import mean, std
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

features = \
    [
        "ssr_south",
        "ssr_middle",
        "ssr_north",
        "str_south",
        "str_middle",
        "str_north",
        "t2m_north",
        "t2m_middle",
        "t2m_south",
        "summertime",
        "hour-0",
        "hour-12",
        "hour-1",
        "hour-2",
        "hour-3",
        "hour-4",
        "hour-5",
        "hour-6",
        "hour-7",
        "hour-8",
        "hour-9",
        "hour-10",
        "hour-11",
        "hour-13",
        "hour-14",
        "hour-15",
        "hour-16",
        "hour-17",
        "hour-18",
        "hour-19",
        "hour-20",
        "hour-21",
        "hour-22",
        "hour-23",
        "lag_0",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_5",
        "lag_6",
        "lag_7"
    ]

category_weekdays = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday"
]

category_hours = [
    "hour-0",
    "hour-12",
    "hour-1",
    "hour-2",
    "hour-3",
    "hour-4",
    "hour-5",
    "hour-6",
    "hour-7",
    "hour-8",
    "hour-9",
    "hour-10",
    "hour-11",
    "hour-13",
    "hour-14",
    "hour-15",
    "hour-16",
    "hour-17",
    "hour-18",
    "hour-19",
    "hour-20",
    "hour-21",
    "hour-22",
    "hour-23"
]

category_full = [
    "ssr_south",
    "ssr_middle",
    "ssr_north",
    "str_south",
    "str_middle",
    "str_north",
    "t2m_north",
    "t2m_middle",
    "t2m_south",
    "summertime",
    "hour",
    "lag_0",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_5",
    "lag_6",
    "lag_7"

]

scope_var = "solar_with_curtailment"



if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # DATAPATH to set
    data = pd.read_pickle("solar_0.pkl")
    X_train = data[features][192:-8760]
    y_train = data[scope_var][192:-8760]

    print("Sample Count :" + str(y_train.shape[0]))

    names = X_train.columns


    def get_weekdays(row):
        """
        Makes the Weekdays categorical
        :param row:
        :return:
        """
        count = 0
        for c in category_weekdays:

            if row[c] == 1:
                return count
            else:
                count = count + 1


    def get_hours(row):
        """
        Makes the hours categorical
        :param row:
        :return:
        """
        count = 0
        for c in category_hours:

            if row[c] == 1:
                return count
            else:
                count = count + 1


    X_train["weekday"] = X_train.apply(get_weekdays, axis=1).astype("category")
    X_train["hour"] = X_train.apply(get_hours, axis=1).astype("category")

    # define dataset
    X, y = X_train[category_full], y_train
    feature_to_score = []
    fs_sets = []

    for k in range(2, len(category_full) + 1):

        rfe = RFE(estimator=DecisionTreeRegressor(random_state=42), n_features_to_select=k)
        # fitting
        rfe.fit(X, y)
        # summarize all features
        tmp_set = []

        for i in range(X.shape[1]):
            print('Column: %s, Selected %s, Rank: %.3f' % (category_full[i], rfe.support_[i], rfe.ranking_[i]))
            if rfe.support_[i] == True:
                tmp_set.append(category_full[i])

        fs_sets.append([k, tmp_set])

        model = DecisionTreeRegressor(random_state=42)
        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
        score = cross_val_score(model, X[tmp_set], y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1,
                                error_score='raise',
                                verbose=1)

        feature_to_score.append((k, score.mean() * -1))

    result = {
        "fs_to_scores": feature_to_score,
        "fs_sets": fs_sets
    }

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
