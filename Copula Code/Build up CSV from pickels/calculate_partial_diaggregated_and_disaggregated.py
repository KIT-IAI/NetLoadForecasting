import math
import pickle
import numpy as np
from matplotlib import pyplot as plt
import properscoring as ps
import pandas as pd

import glob

from scipy.interpolate import interpolate



def gaussian(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp((1 / 2) * -np.power(x - mean, 2) / (np.power(std, 2)))



def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


if __name__ == '__main__':

    # path to pickels of the disaggregated strategy
    folder ="path/*.pickle"
    # path to pickels of the disaggregated strategy
    folder_part_disaggregatedly =  "path/*.pickle"


    list_of_dicts = []

    crps_disaggregated = []
    crps_part_disaggregated = []
    crps_d = []

    log_loss_disaggregated = []
    log_loss_part_disaggregated = []


    id_part_disaggregated = []
    id_disaggregated = []

    #
    # Path to an csv which contains the ground truth values of the mismatch for traing (for example the pickel of an direct forecast)
    #
    mismatch_df = pd.read_pickle("path.pkl")


    print("Aggregated")

    for x in glob.glob(folder):
        with open(x, "rb") as f:
            dict_res = pickle.load(f)


            xnew = np.arange(-200, 200, 0.1)
            pdf = dict_res["res"][0](xnew)    # use interpolation function returned by `interp1d`
            cdf = dict_res["res"][1](xnew)/np.max(dict_res["res"][1](xnew))
            mismatch = dict_res["gt_mismatch"]
            plt.plot(xnew, pdf, '-')

            #
            #Debug plots
            #
            plt.axvline(x=mismatch, color='r')
            plt.title(dict_res["id"])
            #plt.show()

            cdf_function = interpolate.interp1d(np.arange(-200, 200, 0.1), cdf, bounds_error=False, fill_value=(float(0.0),float(1.0)))
            pdf_function = interpolate.interp1d(np.arange(-200, 200, 0.1), pdf, bounds_error=False, fill_value=(float(0.0),float(1.0)))


            crps_disaggregated.append(ps.crps_quadrature(mismatch, cdf_function, -200, 200, tol=1e-2))
            log_loss_disaggregated.append(-math.log(pdf_function(mismatch)))
            id_disaggregated.append(dict_res["id"])



    print("Part_disaggregated")

    i = 0
    for x in glob.glob(folder_part_disaggregatedly):
        with open(x, "rb") as f:
            dict_res = pickle.load(f)


            xnew = np.arange(-200, 200, 0.1)
            pdf = dict_res["res"][0](xnew)    # use interpolation function returned by `interp1d`
            cdf = dict_res["res"][1](xnew)
            mismatch = dict_res["gt_mismatch"]
            plt.plot(xnew, pdf, '-')
            cdf_function = interpolate.interp1d(np.arange(-200, 200, 0.1), cdf, bounds_error=False,
                                                fill_value=(float(0.0), float(1.0)))
            pdf_function = interpolate.interp1d(np.arange(-200, 200, 0.1), pdf, bounds_error=False,
                                                fill_value=(float(0.0), float(1.0)))


            #
            #Debug plots
            #
            plt.axvline(x=mismatch, color='r')
            plt.title(dict_res["id"])
            #plt.show()

            crps_value = ps.crps_quadrature(mismatch,cdf_function,-200,200,tol=1e-2)
            id_part_disaggregated.append(dict_res["id"])
            log_loss_part_disaggregated.append(-math.log(pdf_function(mismatch)))
            crps_part_disaggregated.append(crps_value)
            print(i ," : " ,crps_value)
            i = i+1



    print("Disaggregated : " + str(np.array(crps_disaggregated).mean()))


    print("Part disaggregated: ",np.array(crps_part_disaggregated).mean())



    crps_part_disaggregated_df = pd.DataFrame({'id' : id_part_disaggregated, "crps" :crps_part_disaggregated, 'log_loss' :log_loss_part_disaggregated})
    crps_disaggregated_df = pd.DataFrame({'id' : id_disaggregated, "crps" :crps_disaggregated, 'log_loss' :log_loss_disaggregated})


    #
    # Save the results and sort
    #
    crps_part_disaggregated_df.sort_values(by="id", ascending=False).to_csv("crps_part_disaggregated.csv")
    crps_disaggregated_df.sort_values(by="id", ascending=False).to_csv("crps_disaggregated.csv")





