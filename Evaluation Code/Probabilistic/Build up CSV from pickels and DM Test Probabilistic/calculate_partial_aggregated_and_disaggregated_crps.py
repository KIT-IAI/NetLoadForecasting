import glob
import math
import pickle

import numpy as np
import pandas as pd
import properscoring as ps
from scipy.interpolate import interpolate

if __name__ == '__main__':

    # path to pickels of the disaggregated strategy
    folder = "path/*.pickle"
    # path to pickels of the partial aggregated strategy
    folder_part_aggregated = "path/*.pickle"

    list_of_dicts = []

    crps_disaggregated = []
    crps_part_aggregated = []
    crps_d = []

    log_loss_disaggregated = []
    log_loss_part_aggregated = []

    id_part_aggregated = []
    id_disaggregated = []

    print("Disaggregated")

    for x in glob.glob(folder):
        with open(x, "rb") as f:
            dict_res = pickle.load(f)

            xnew = np.arange(-200, 200, 0.1)
            pdf = dict_res["res"][0](xnew)  # use interpolation function returned by `interp1d`
            cdf = dict_res["res"][1](xnew) / np.max(dict_res["res"][1](xnew))
            mismatch = dict_res["gt_mismatch"]
            cdf_function = interpolate.interp1d(np.arange(-200, 200, 0.1), cdf, bounds_error=False,
                                                fill_value=(float(0.0), float(1.0)))
            pdf_function = interpolate.interp1d(np.arange(-200, 200, 0.1), pdf, bounds_error=False,
                                                fill_value=(float(0.0), float(0.0)))
            crps_disaggregated.append(ps.crps_quadrature(mismatch, cdf_function, -200, 200, tol=1e-2))
            log_loss_disaggregated.append(-math.log(pdf_function(mismatch)))
            id_disaggregated.append(dict_res["id"])

    print("Part_aggregated")

    i = 0
    for x in glob.glob(folder_part_aggregated):
        with open(x, "rb") as f:
            dict_res = pickle.load(f)

            xnew = np.arange(-200, 200, 0.1)
            pdf = dict_res["res"][0](xnew)  # use interpolation function returned by `interp1d`
            cdf = dict_res["res"][1](xnew)
            mismatch = dict_res["gt_mismatch"]
            cdf_function = interpolate.interp1d(np.arange(-200, 200, 0.1), cdf, bounds_error=False,
                                                fill_value=(float(0.0), float(1.0)))
            pdf_function = interpolate.interp1d(np.arange(-200, 200, 0.1), pdf, bounds_error=False,
                                                fill_value=(float(0.0), float(0.0)))

            crps_value = ps.crps_quadrature(mismatch, cdf_function, -200, 200, tol=1e-2)
            id_part_aggregated.append(dict_res["id"])
            log_loss_part_aggregated.append(-math.log(pdf_function(mismatch)))
            crps_part_aggregated.append(crps_value)

    print("Disaggregated : " + str(np.array(crps_disaggregated).mean()))

    print("Part aggregated: ", np.array(crps_part_aggregated).mean())

    crps_part_aggregated_df = pd.DataFrame(
        {'id': id_part_aggregated, "crps": crps_part_aggregated, 'log_loss': log_loss_part_aggregated})
    crps_disaggregated_df = pd.DataFrame(
        {'id': id_disaggregated, "crps": crps_disaggregated, 'log_loss': log_loss_disaggregated})

    #
    # Save the results and sort
    #
    crps_part_aggregated_df = crps_part_aggregated_df.sort_values(by="id")
    crps_part_aggregated_df.to_csv("crps_part_aggregated.csv")

    crps_disaggregated_df = crps_disaggregated_df.sort_values(by="id")
    crps_disaggregated_df.to_csv("crps_disaggregated.csv")

    #
    # Just use the resulting Df on the modified dm_test in this folder to get significance
    # dm_test_crps(None, error_list_1, error_list_2, None, None, None):
    #
