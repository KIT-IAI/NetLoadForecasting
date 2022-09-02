import argparse
import glob
import os.path
import pathlib
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.integrate import quad
from scipy.stats import norm, beta
from statsmodels.distributions.copula.api import FrankCopula


def _get_normal_pdf(mean, std, x):
    return norm.pdf(loc=mean, scale=std, x=x)


def _get_normal_cdf(mean, std, x):
    return norm.cdf(loc=mean, scale=std, x=x)


#
# x series of samples, y series of samples
# actual only frank is implemented
#
def _fit_copula(x, y, fam="frank"):
    if fam == "frank":
        c = FrankCopula()
        data = np.column_stack((x, y))
        theta = c.fit_corr_param(data)

        return theta
        # Parameters are estimated and set
    else:
        raise NotImplementedError("not implemented")


def _get_copula_density(u, v, fam, params):
    """
    returns the coupula density (u,v) for given copula family
    :param u:
    :param v:
    :param fam:
    :param params:
    :return: copula density (u,v)
    """
    if fam == "frank":
        theta = params[0]

        # Copula Density according to
        # http://www.crest.fr/ckfinder/userfiles/files/pageperso/fermania/chapter-book-copula-density-estimation.pdf

        num = theta * (1 - np.exp(-theta)) * np.exp(-theta * (u + v))
        den = np.square((1 - np.exp(-theta)) - (1 - np.exp(-theta * u)) * (1 - np.exp(-theta * v)))

        res = num / den
        return res

    else:
        raise NotImplementedError("not implemented")


def get_densitys(pdf_a, cdf_a, pdf_b, cdf_b, params_copular, copular_fam="frank", ignore_a=False, ignore_b=False,
                 plot=True, plt_title="plot"):
    """
    given the parameters it returns the densitys by using the copula
    :param pdf_a:
    :param cdf_a:
    :param pdf_b:
    :param cdf_b:
    :param params_copular:
    :param copular_fam:
    :param ignore_a:
    :param ignore_b:
    :param plot:
    :param plt_title:
    :return: pdf and cdf
    """
    if ignore_a:

        pdf = pdf_b
        cdf = cdf_b

        if plot:
            xnew = np.arange(-200, 200, 0.1)
            ynew = pdf(xnew)  # use interpolation function returned by `interp1d`
            plt.plot(xnew, ynew, '-')
            plt.title(plt_title)
            plt.show()

        return pdf, cdf

    if ignore_b:
        pdf = pdf_a
        cdf = cdf_a

        if plot:
            xnew = np.arange(-200, 200, 0.1)
            ynew = pdf(xnew)  # use interpolation function returned by `interp1d`
            plt.plot(xnew, ynew, '-')
            plt.title(plt_title)
            plt.show()

        return pdf, cdf

    density = CombinedDensity(
        pdf_a,
        pdf_b,
        cdf_a,
        cdf_b,
        copular_fam,
        params_copular
    )

    pmf_array = np.asarray([density.get_density_value(z)[0] for z in np.arange(-200, 200, 0.1)])
    pdf = interpolate.interp1d(np.arange(-200, 200, 0.1), pmf_array, bounds_error=False, fill_value=float(0.0))

    cdf_array = np.cumsum(pmf_array * 0.1)

    cdf = interpolate.interp1d(np.arange(-200, 200, 0.1), cdf_array, bounds_error=False, fill_value=(0, 1))

    if plot:
        plt.plot(np.arange(-200, 200, 0.1), cdf_array, 'r--')
        xnew = np.arange(-200, 200, 0.1)
        ynew = pdf(xnew)  # use interpolation function returned by `interp1d`
        plt.plot(np.arange(-200, 200, 0.1), pmf_array, 'o', xnew, ynew, '-')
        plt.title(plt_title)
        plt.show()

    return pdf, cdf


def get_resulting_density(i, demand_df, supply_df):
    """
    Get the resulting densitys given the dataframe and an index
    :param i: index
    :param demand_df: demand df
    :param supply_df: supply df
    :return: pdf and cdf regarding index i
    """

    beta_supply_pdf = lambda d: beta.pdf(d, supply_df["concentration1"][i], supply_df["concentration0"][i])
    beta_supply_cdf = lambda d: beta.cdf(d, supply_df["concentration1"][i], supply_df["concentration0"][i])
    scaleparam_supply = supply_df["scaleparam"][i]

    pdf_array_demand_supply, cdf_array_demand_supply = get_densitys(
        lambda a: _get_normal_pdf(demand_df["mean"][i], demand_df["std"][i], a),
        lambda c: _get_normal_cdf(demand_df["mean"][i], demand_df["std"][i], c),
        lambda b: (1 / scaleparam_supply) * beta_supply_pdf((-b / scaleparam_supply)),
        lambda d: 1 - beta_supply_cdf(-d / scaleparam_supply),
        [theta_demand_supply],
        plt_title="Demand - Supply " + str(i)

    )

    print(str(i) + " gets saved")

    mismatch = demand_df["demand"][i] - supply_df["supply_re"][i]

    #
    # Contains pdf, cdf, id (8760 for each hour of the year) and the ground truth value of the mismatch
    #

    dict_to_save = {
        'res': [pdf_array_demand_supply, cdf_array_demand_supply],
        'id': i,
        'gt_mismatch': mismatch
    }

    with open(model_path + "/results_demand_supply/" + str(i) + '.pickle', 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(str(i) + "saved")

    return pdf_array_demand_supply, cdf_array_demand_supply


class CombinedDensity:
    """
    Class that handels the copula and calculates
    """

    def __init__(self, pdf_x, pdf_y, cdf_x, cdf_y, copula_fam, params):
        self.pdf_x = pdf_x
        self.pdf_y = pdf_y
        self.cdf_x = cdf_x
        self.cdf_y = cdf_y

        tmp_index = np.arange(-200, 200, 0.1)
        max_idx = np.argmax([pdf_x(x) for x in np.arange(-200, 200, 0.1)])
        max_idy = np.argmax([pdf_y(x) for x in np.arange(-200, 200, 0.1)])
        self.max_x = tmp_index[max_idx]
        self.max_y = tmp_index[max_idy]

        self.copula_fam = copula_fam
        self.params = params

    def _integral_part(self, z, t):
        cdf_x_value = self.cdf_x(z - t)
        cdf_y_value = self.cdf_y(t)

        pdf_x_value = self.pdf_x(z - t)
        pdf_y_value = self.pdf_y(t)

        c = _get_copula_density(cdf_x_value, cdf_y_value, "frank", self.params)

        return c * pdf_x_value * pdf_y_value

    def get_density_value(self, z):
        f = lambda t: self._integral_part(z, t)

        point_x = np.array(
            [z - self.max_x - 0.1, z - self.max_x - 0.05, z - self.max_x, z - self.max_x + 0.05, z - self.max_x + 0.1])
        point_y = np.array([z - self.max_y - 0.1, self.max_y - 0.05, self.max_y, self.max_y + 0.05, self.max_y + 0.1])

        a = np.concatenate((point_x, point_y))
        values, counts = np.unique(a, return_counts=True)

        f_a_b = quad(f, -200, 200, limit=100, points=values)

        return f_a_b


#######################################################################################################################

#
# Main part here the execution happens
#





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser of parameters for the coupla execution")
    parser.add_argument('-lb', type=int, help="Lower bound")
    parser.add_argument('-hb', type=int, help="Higher bound")
    parser.add_argument('-t', type=int, help="Number of threads")
    parser.add_argument('-folder', type=pathlib.Path, help="Path of the resulting pickels")

    args = parser.parse_args()

    #
    # This paths need to be fitted if other structure is used
    #

    #
    # Need to have data of gt mismatch
    #

    gt_path = "Data/mismatch_0.pkl"

    #
    # there need to be a folder "model_path + "/results_demand_supply/". There the resulting pickel files get saved.
    #

    model_path = str(args.folder)

    # Demand

    demand_path = glob.glob(os.path.join(model_path, "subresults/demand/*.pkl"))[0]

    # Supplier

    supply_path = glob.glob(os.path.join(model_path, "subresults/supply_re/*.pkl"))[0]

    mismatch_df = pd.read_pickle(gt_path)
    demand_df = pd.read_pickle(demand_path)
    supply_df = pd.read_pickle(supply_path)

    #
    # samples used to fit the copula is a subset containing full years only used for training purpose. Therefore the validation date is not used.
    #

    demand_samples = mismatch_df["demand"][0:35040]

    supply_samples = mismatch_df["supply_re"][0:35040]

    #
    # To fit the copula
    #

    theta_demand_supply = _fit_copula(demand_samples, -supply_samples)

    print("Every Copular")

    print(theta_demand_supply)

    res = None

    # multithread solution

    with Pool() as p:
        res = p.starmap(get_resulting_density,
                        [(i, demand_df, supply_df) for i in range(args.lb, args.hb)])

    print("done")
