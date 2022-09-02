import math

import numpy as np
import pandas as pd
import properscoring as ps
from scipy.stats import beta

if __name__ == '__main__':

    df = pd.read_pickle("path.pkl")
    id = []
    crps = []
    log_loss = []

    for i in range(8760):

        beta_solar_pdf = lambda d: beta.pdf(d, df["concentration1"][i], df["concentration0"][i])
        beta_solar_cdf = lambda d: beta.cdf(d, df["concentration1"][i], df["concentration0"][i])

        scaleparam = 99

        dist_pdf = lambda b: (1 / scaleparam) * beta_solar_pdf((b / scaleparam))
        dist_cdf = lambda d: beta_solar_cdf(d / scaleparam)

        tmp = ps.crps_quadrature(df["solar_with_curtailment"][i], dist_cdf, -200, 200, tol=1e-2)

        print(i, tmp)
        id.append(i)
        crps.append(tmp)

        if (dist_pdf(df["solar_with_curtailment"][i]) != 0):
            log_loss.append(-math.log(dist_pdf(df["solar_with_curtailment"][i])))
        else:
            log_loss.append(-math.log(dist_pdf(df["solar_with_curtailment"][i] + 0.1)))

    df = pd.DataFrame(
        {'id': id, "crps": crps, 'log_loss': log_loss}).sort_values(by="id", ascending=True)

    df.to_csv("results.csv", index=False)
    print("MEAN Logloss DIRECT: ", np.asarray(log_loss).mean())
    print("MEAN CRPS DIRECT: ", np.asarray(crps).mean())
