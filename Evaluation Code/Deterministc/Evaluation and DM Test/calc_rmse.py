import math

import numpy as np
import pandas as pd

#
# Can be used for ever result file
#

if __name__ == '__main__':
    #
    # use with var and corresponding path
    #

    path = "path.pkl"
    #
    # hat variable for example mismatch in this case
    #
    hat_variable = "mismatch_demand_renewable_hat"
    #
    # variable for example mismatch in this case
    #
    variable = "mismatch_demand_renewable"

    df = pd.read_pickle(path)

    y_actual = df[variable]
    y_predicted = df[hat_variable]
    MSE = np.square(np.subtract(y_actual, y_predicted)).mean()
    RMSE = math.sqrt(MSE)

    print(RMSE)
