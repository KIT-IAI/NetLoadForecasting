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
from scipy.stats import norm, gamma, beta
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


def get_resulting_density(i, demand_df, solar_df, onshore_df, offshore_df, ror_df, supply_df):
    """
    Get the resulting densitys given the dataframe and an index
    :param i: index
    :param demand_df: demand dataframe
    :param solar_df: solar dataframe
    :param onshore_df: onshore dataframe
    :param offshore_df: offshore dataframe
    :param ror_df: ror dataframe
    :param supply_df: supply dataframe here only for calculate the gt mismatch
    :return: pdf , cdf of index i
    """

    pdf_array_demand_solar, cdf_array_demand_solar = None, None
    #
    # Check Solar
    #
    # Filter zeroes as they are difficult due to numerical handling
    #

    exp = solar_df["concentration1"][i] / (solar_df["concentration1"][i] + solar_df["concentration0"][i])

    exp = exp * 99

    if exp <= 0.3:

        pdf_array_demand_solar, cdf_array_demand_solar = get_densitys(
            lambda a: _get_normal_pdf(demand_df["mean"][i], demand_df["std"][i], a),
            lambda c: _get_normal_cdf(demand_df["mean"][i], demand_df["std"][i], c),
            None,
            None,
            [theta_demand_solar],
            plt_title="Demand - Solar : " + str(i),
            ignore_b=True

        )
    else:
        beta_solar_pdf = lambda d: beta.pdf(d, solar_df["concentration1"][i], solar_df["concentration0"][i])
        beta_solar_cdf = lambda d: beta.cdf(d, solar_df["concentration1"][i], solar_df["concentration0"][i])
        scaleparam_solar = solar_df["scaleparam"][i]

        pdf_array_demand_solar, cdf_array_demand_solar = get_densitys(
            lambda a: _get_normal_pdf(demand_df["mean"][i], demand_df["std"][i], a),
            lambda c: _get_normal_cdf(demand_df["mean"][i], demand_df["std"][i], c),
            lambda b: (1 / scaleparam_solar) * beta_solar_pdf((-b / scaleparam_solar)),
            lambda d: 1 - beta_solar_cdf(-d / scaleparam_solar),
            [theta_demand_solar],
            plt_title="Demand - Solar : " + str(i),

        )

    #
    #   Demand Solar and Onshore
    #

    scale_onshore = 1 / onshore_df["rate"][i]
    a = onshore_df["concentration"][i]

    pdf_array_demand_solar_on, cdf_array_demand_solar_on = get_densitys(
        lambda a: pdf_array_demand_solar(a),
        lambda c: cdf_array_demand_solar(c),
        lambda b: gamma.pdf(-b, a, scale=scale_onshore),
        lambda d: 1 - gamma.cdf(-d, a, scale=scale_onshore),
        [theta_demand_solar_on],
        plt_title="Demand - Solar - Onshore"
    )

    #
    #   Demand Solar Onshore and Offshore
    #

    beta_offshore_pdf = lambda d: beta.pdf(d, offshore_df["concentration1"][i], offshore_df["concentration0"][i])
    beta_offshore_cdf = lambda d: beta.cdf(d, offshore_df["concentration1"][i], offshore_df["concentration0"][i])

    # print(offshore_df["concentration1"], offshore_df["concentration0"][i], offshore_df["scaleparam"][i])

    scaleparam = offshore_df["scaleparam"][i]

    pdf_array_demand_solar_on_off, cdf_array_demand_solar_on_off = get_densitys(
        lambda a: pdf_array_demand_solar_on(a),
        lambda c: cdf_array_demand_solar_on(c),
        lambda b: (1 / scaleparam) * beta_offshore_pdf((-b / scaleparam)),
        lambda d: 1 - beta_offshore_cdf(-d / scaleparam),
        [theta_demand_solar_on_off],
        plt_title="Demand - Solar - Onshore - Offshore"
    )

    #
    # Demand Solar Onshore Offshore and Ror
    #

    beta_ror_pdf = lambda d: beta.pdf(d, ror_df["concentration1"][i], ror_df["concentration0"][i])
    beta_ror_cdf = lambda d: beta.cdf(d, ror_df["concentration1"][i], ror_df["concentration0"][i])

    scaleparam = ror_df["scaleparam"][i]

    pdf_array_demand_solar_on_off_ror, cdf_array_demand_solar_on_off = get_densitys(

        lambda a: pdf_array_demand_solar_on_off(a),
        lambda c: cdf_array_demand_solar_on_off(c),
        lambda b: (1 / scaleparam) * beta_ror_pdf((-b / scaleparam)),
        lambda d: 1 - beta_ror_cdf(-d / scaleparam),
        [theta_demand_solar_on_off_ror],
        plt_title="Demand - Solar - Onshore - Offshore -Ror :" + str(i)

    )

    print(str(i) + " gets saved")

    mismatch = demand_df["demand"][i] - supply_df["supply_re"][i]

    dict_to_save = {
        'res': [pdf_array_demand_solar_on_off_ror, cdf_array_demand_solar_on_off],
        'id': i,
        'gt_mismatch': mismatch
    }

    with open(model_path + "/results_disaggregated/" + str(i) + '.pickle', 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(str(i) + "saved")

    return pdf_array_demand_solar_on_off_ror, cdf_array_demand_solar_on_off


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

        self.estimated_target = self.max_x - self.max_y

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

    model_path = str(args.folder)

    #
    # This path need to be fitted if other structure is used
    #
    # Demand

    demand_path = glob.glob(os.path.join(model_path, "subresults/demand/*.pkl"))[0]

    mismatch_df = pd.read_pickle(gt_path)

    demand_df = pd.read_pickle(demand_path)

    #
    # This paths need to be fitted if other structure is used
    #
    # Supplier
    supply_path = glob.glob(os.path.join(model_path, "subresults/supply_re/*.pkl"))[0]
    solar_path = glob.glob(os.path.join(model_path, "subresults/solar_with_curtailment/*.pkl"))[0]
    offshore_path = glob.glob(os.path.join(model_path, "subresults/wind_agg_off_with_curtailment/*.pkl"))[0]
    onshore_path = glob.glob(os.path.join(model_path, "subresults/onwind_with_curtailment/*.pkl"))[0]
    ror_path = glob.glob(os.path.join(model_path, "subresults/ror_with_curtailment/*.pkl"))[0]

    mismatch_df = pd.read_pickle(gt_path)

    supply_df = pd.read_pickle(supply_path)
    demand_df = pd.read_pickle(demand_path)
    solar_df = pd.read_pickle(solar_path)
    onshore_df = pd.read_pickle(onshore_path)
    offshore_df = pd.read_pickle(offshore_path)
    ror_df = pd.read_pickle(ror_path)

    #
    # samples used to fit the copula is a subset containing full years
    # only used for training purpose. Therefore the validation date is not used.
    #

    demand_samples = mismatch_df["demand"][0:35040]
    solar_samples = mismatch_df["solar"][0:35040] + mismatch_df["curtailment-solar"][0:35040]
    offshore_samples = mismatch_df['offwind-ac'][0:35040] + mismatch_df['offwind-dc'][0:35040] + mismatch_df[
                                                                                                     'curtailment-offwind-ac'][
                                                                                                 0:35040] + mismatch_df[
                                                                                                                'curtailment-offwind-dc'][
                                                                                                            0:35040]
    onshore_samples = mismatch_df['onwind'][0:35040] + mismatch_df['curtailment-onwind'][0:35040]
    ror_samples = mismatch_df['ror'][0:35040] + mismatch_df['curtailment-ror'][0:35040]

    #
    # To fit the copula the data need to be obtained
    #

    demand_minus_solar = mismatch_df["demand"][0:35040] - mismatch_df["solar"][0:35040] + mismatch_df[
                                                                                              "curtailment-solar"][
                                                                                          0:35040]

    demand_minus_solar_minus_on = demand_minus_solar - onshore_samples

    demand_minus_solar_minus_on_minus_off = demand_minus_solar_minus_on - offshore_samples

    demand_minus_solar_minus_on_minus_off_minus_ror = demand_minus_solar_minus_on_minus_off - ror_samples

    #
    # samples used to fit the copula is a subset containing full years
    # only used for training purpose. Therefore the validation date is not used.
    #

    theta_demand_solar = _fit_copula(demand_samples, -solar_samples)

    theta_demand_solar_on = _fit_copula(demand_minus_solar, -onshore_samples)

    theta_demand_solar_on_off = _fit_copula(demand_minus_solar_minus_on, -offshore_samples)

    theta_demand_solar_on_off_ror = _fit_copula(demand_minus_solar_minus_on_minus_off, -ror_samples)

    print("Print Every Copular:")

    print(theta_demand_solar, theta_demand_solar_on, theta_demand_solar_on_off, theta_demand_solar_on_off_ror)

    res = None

    # multithread solution

    with Pool() as p:
        res = p.starmap(get_resulting_density,
                        [(i, demand_df, solar_df, onshore_df, offshore_df, ror_df, supply_df) for i in
                         range(args.lb, args.hb)])

    print("done")
