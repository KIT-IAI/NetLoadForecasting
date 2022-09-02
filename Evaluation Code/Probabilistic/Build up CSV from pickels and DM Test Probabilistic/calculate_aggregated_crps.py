import math

import numpy as np
import pandas as pd
import properscoring as ps
from scipy.stats import beta

if __name__ == '__main__':

    #
    # this needs to be the pkl file containing the results on test of the best fully aggregated model
    #
    df = pd.read_pickle("path.pkl")

    id = []
    crps = []
    log_loss = []
    for i in range(8760):
        beta_mismatch_cdf = lambda d: beta.cdf((d - (-115)) / (93.75 + 115), df["concentration1"][i],
                                               df["concentration0"][i])
        beta_mismatch_pdf = lambda d: (1 / (93.75 + 115)) * (
            beta.pdf((d - (-115)) / (93.75 + 115), df["concentration1"][i], df["concentration0"][i]))
        tmp = ps.crps_quadrature(df["mismatch_demand_renewable"][i], beta_mismatch_cdf, -200, 200, tol=1e-2)
        print(i, tmp)
        id.append(i)
        crps.append(tmp)
        log_loss.append(-math.log(beta_mismatch_pdf(df["mismatch_demand_renewable"][i])))

    df = pd.DataFrame(
        {'id': id, "crps": crps, 'log_loss': log_loss})

    df.to_csv("direct_results_prob.csv")

    print("MEAN Logloss DIRECT: ", np.asarray(log_loss).mean())
    print("MEAN CRPS DIRECT: ", np.asarray(crps).mean())
