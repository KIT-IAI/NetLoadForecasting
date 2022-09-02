import math

import numpy as np
import pandas as pd
import properscoring as ps
from scipy.stats import gamma

if __name__ == '__main__':

    df = pd.read_pickle("path.pkl")
    id = []
    crps = []
    log_loss = []

    for i in range(8760):
        scale_onshore = 1 / df["rate"][i]
        a = df["concentration"][i]

        dist_pdf = lambda b: gamma.pdf(b, a, scale=scale_onshore)
        dist_cdf = lambda d: gamma.cdf(d, a, scale=scale_onshore)

        tmp = ps.crps_quadrature(df["onwind_with_curtailment"][i], dist_cdf, -200, 200, tol=1e-2)

        print(i, tmp)
        id.append(i)
        crps.append(tmp)
        log_loss.append(-math.log(dist_pdf(df["onwind_with_curtailment"][i])))

    df = pd.DataFrame(
        {'id': id, "crps": crps, 'log_loss': log_loss}).sort_values(by="id", ascending=True)

    df.to_csv("results.csv", index=False)

    print("MEAN Logloss DIRECT: ", np.asarray(log_loss).mean())
    print("MEAN CRPS DIRECT: ", np.asarray(crps).mean())
