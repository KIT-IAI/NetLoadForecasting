import math

import numpy as np
import pandas as pd
import properscoring as ps
from scipy.stats import norm

if __name__ == '__main__':

    df = pd.read_pickle("path.pkl")

    id = []
    crps = []
    log_loss = []
    for i in range(8760):
        dist_pdf = lambda a: norm.pdf(a, loc=df["mean"][i], scale=df["std"][i])
        dist_cdf = lambda a: norm.cdf(a, loc=df["mean"][i], scale=df["std"][i])

        tmp = ps.crps_gaussian(df["demand"][i], mu=df["mean"][i], sig=df["std"][i])
        print(i, tmp)
        id.append(i)
        crps.append(tmp)
        log_loss.append(-math.log(dist_pdf(df["demand"][i])))

    df = pd.DataFrame(
        {'id': id, "crps": crps, 'log_loss': log_loss}).sort_values(by="id", ascending=True)

    df.to_csv("results.csv", index=False)

    print("MEAN Logloss DIRECT: ", np.asarray(log_loss).mean())
    print("MEAN CRPS DIRECT: ", np.asarray(crps).mean())
