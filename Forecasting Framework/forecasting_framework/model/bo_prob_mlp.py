import os
import random as python_random
import sys

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from forecasting_framework.model.model import Model
from forecasting_framework.model.model_prob_mlp import ProbMlp
from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
)
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

sys.path.append(".")

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(42)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.compat.v1.set_random_seed(42)


# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/


def negative_loglikelihood(targets, estimated_distribution):
    '''
    Loss function negative log
    '''

    return -estimated_distribution.log_prob(targets)


class CVTuner(kt.engine.tuner.Tuner):
    '''
    Runs of the bayesian optimisation, combined with the four fold. Therefore this custom tuner is used.
    Same tuner as used for the simple approach but there it uses the functionality of multiple Inputs.
    '''

    bo_epochs = 1000
    bo_patience = 15

    def run_trial(self, trial, x, y, batch_size=128):
        print("Run TrialID : " + str(trial.trial_id) + " started.")
        folds = 4
        cv = model_selection.TimeSeriesSplit(n_splits=folds, test_size=8760)
        val_losses = []

        count = 1
        for train_indices, test_indices in cv.split(x):
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            tf.keras.backend.clear_session()

            model = self.hypermodel.build(trial.hyperparameters)
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss=negative_loglikelihood)
            print("CV " + str(count) + " out of " + str(folds) + " started.")
            model.summary()
            model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test),
                      callbacks=[
                          tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                                           patience=self.bo_patience, min_delta=0.001,
                                                           restore_best_weights=True)],
                      epochs=self.bo_epochs, verbose=1)

            val_losses.append(model.evaluate(x_test, y_test))

            print("CV " + str(count) + " out of " + str(folds) + " ended.")
            print("Val_losses : " + str(val_losses))
            count = count + 1

        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
        self.save_model(trial.trial_id, model)


class BoProbMLP(Model):
    """
    Performs bayesian optimisation for the network in the training method, then applies the model in the test method.
    """

    #
    # Variables
    #
    data = None
    to_predict_features = None
    scaler_forecast_varaibles = None
    scaler_features = None
    scaler_lag = None
    localmodel = None
    hyperparameters_list = None
    train_epochs = 300
    train_patience = 20

    def __init__(self, data_path,
                 modelparams_dict, features, forecast_variables, modelname, train, folder, name, plot_diagnostic=False,
                 save_diagnostics=False):
        Model.__init__(self, data_path=data_path, modelparams=modelparams_dict,
                       features=features,
                       forecast_variables=forecast_variables, folder=folder, plot_diagnostic=plot_diagnostic,
                       train=train,
                       save_diagnostics=save_diagnostics, name=name, modelname="nn")

    def execute_train(self):
        """
        Execution of the training of the model and the hyperparameter optimization using bayesian optimisation.
        """

        if Model.test_path(self, self.name, self.folder) and not self.modelparams_dict["testset"]:
            print("Model already exists :" + self.name + "/" + self.folder)
            return
        else:
            print("Model starts :" + self.name + "/" + self.folder)

        data_sets_path = Model._folder
        load_data_path = os.path.join(data_sets_path, self.data_path)

        data = pd.read_pickle(load_data_path)

        # throw away the first 192 8 * 24 for lag 0 to lag 7
        x_with_lag = data[self.features][192:]
        y = data[self.forecast_variables][192:]

        print("loaded")

        # Seperate lag features and
        # fit scaler on val and train data

        train_val = x_with_lag[x_with_lag.index.year <= 2018]

        self.scaler_features = StandardScaler().fit(train_val)
        self.scaler_forecast_varaibles = StandardScaler().fit((y[y.index.year <= 2018]))

        y = y.to_numpy()

        #
        # In order to resolve problems with zero values they are minimal shifted. Furthermore, they are scaled.
        #

        for i in range(y.shape[0]):

            if y[i] == 0 and self.modelparams_dict["dist"] == "gamma":
                y[i] = 0.01

            elif y[i] == 0 and self.modelparams_dict["dist"] == "beta":
                y[i] = (0.01) / self.modelparams_dict["scaleparam"]

            elif self.modelparams_dict["dist"] == "beta_with_min_max":
                y[i] = (y[i] - self.modelparams_dict[
                    "scaleparam_min"]) / (
                               self.modelparams_dict["scaleparam_max"] - self.modelparams_dict["scaleparam_min"])

            elif self.modelparams_dict["dist"] == "beta":
                y[i] = y[i] / self.modelparams_dict["scaleparam"]

        #
        # Scale lag features
        #

        x = self.scaler_features.transform(x_with_lag)

        #
        # Split into train test val
        #

        x_train_val = x[:-8760]
        y_train_val = y[:-8760]

        x_test = x[-8760:]
        y_test = y[-8760:]

        print("Shape Test/Val Size :")
        print(x_train_val.shape)
        print(y_train_val.shape)

        print("Shape Test Size :")
        print(x_test.shape)
        print(y_test.shape)

        features_train_val = x_train_val
        features_test = x_test

        var = ProbMlp(feature_size=features_train_val.shape[1], dist=self.modelparams_dict["dist"])

        # Fixes working directory issues
        working_dir = os.getcwd()
        if working_dir.split("/")[-1] != "forecasting_framework":
            working_dir = working_dir + "/forecasting_framework"

        #
        # BayesianOptimization
        #

        tuner = CVTuner(
            hypermodel=var.get_prob_model,
            oracle=kt.oracles.BayesianOptimization(
                objective=kt.Objective('val_loss', direction="min"),
                max_trials=100, seed=42)
            ,
            directory=str(working_dir) + "/models/baysian_results/" + str(self.forecast_variables[0]),
            project_name=str(self.folder),
            logger=TensorBoardLogger(
                metrics=['val_loss', 'loss'],
                logdir=str(working_dir) + "/models/baysian_results/" + str(self.forecast_variables[0]) + "/" + str(
                    self.folder) + "_tb" + "/logs/hparams"
            )
        )

        tuner.search(features_train_val, y_train_val)
        tuner.search_space_summary()

        print("Done")

        self.to_predict_features = features_test

        best_hp = tuner.get_best_hyperparameters()[0]
        tuner.results_summary()
        self.hyperparameters_list = [x.get_config() for x in tuner.get_best_hyperparameters()]

        # Train the model after bayesian optimisation

        self.localmodel = []

        folds = 4
        cv = model_selection.TimeSeriesSplit(n_splits=folds, test_size=8760)
        count = 1

        for train_indices, test_indices in cv.split(features_train_val):
            # Seed every random variable to ensure consistent training
            np.random.seed(42)
            python_random.seed(42)
            tf.compat.v1.set_random_seed(42)

            tmp_model = var.get_prob_model(best_hp)

            #
            # adam and logloss used
            #

            tmp_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=negative_loglikelihood)

            x_train_fold, x_test_fold = x_train_val[train_indices], x_train_val[test_indices]
            y_train_fold, y_test_fold = y_train_val[train_indices], y_train_val[test_indices]

            print("Training of model " + str(count) + " out of " + str(folds) + " began.")
            history = tmp_model.fit(x_train_fold, y_train_fold, batch_size=128, validation_data=(x_test_fold, y_test_fold),
                                    callbacks=[
                                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                                                         patience=self.train_patience,
                                                                         restore_best_weights=True, min_delta=0.001)
                                    ],
                                    epochs=self.train_epochs, verbose=1)
            self.localmodel.append(tmp_model)

            print("Training of Model " + str(count) + " out of " + str(folds) + " ended.")
            count = count + 1

    def execute_forecast(self):
        """
        Execution of the prediction of the model on the training data. Here the already trained model is used.
        """

        if Model.test_path(self, self.name, self.folder) and not self.modelparams_dict["testset"]:
            print("Model already exists")
            return

        #
        # Get the basic data
        #
        datapath = Model._folder

        testset_path = os.path.join(datapath, self.data_path)

        testset_to_save = pd.read_pickle(testset_path)[-8760:]

        #
        # Prediction and the write-back process for the different distributions
        #

        if self.modelparams_dict["dist"] == "normal":

            results_mean = np.array([x(self.to_predict_features).mean().numpy() for x in self.localmodel])
            results_std = np.array([x(self.to_predict_features).stddev().numpy() for x in
                                    self.localmodel])

            all_models_mean = pd.DataFrame(results_mean.squeeze().T)
            all_models_std = pd.DataFrame(results_std.squeeze().T)
            all_models = pd.concat([all_models_mean, all_models_std], axis=1)

            #
            # Save the Hyperparameters and the result of the single models of the ensemble
            #

            Model.save_pkl(self, all_models, "result_all_models",
                           os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))
            Model.save_csv(self, all_models, "result_all_models",
                           os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))
            Model.save_dict(self, {'Hyperparameters': self.hyperparameters_list}, "Hyperparameters",
                            os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))

            mean = all_models_mean.mean(axis=1).to_numpy()
            std = all_models_std.mean(axis=1).to_numpy()
            testset_to_save["mean"] = mean
            testset_to_save["std"] = std

            #
            # Get a point forecast
            #
            hat_variable = str(self.forecast_variables[0]) + "_hat"
            testset_to_save[hat_variable] = mean



        elif self.modelparams_dict["dist"] == "beta_with_min_max":

            results_concentration = np.array(
                [x(self.to_predict_features).parameters["distribution"].concentration0.numpy() for x in
                 self.localmodel])
            results_rate = np.array(
                [x(self.to_predict_features).parameters["distribution"].concentration1.numpy() for x in
                 self.localmodel])

            all_models_concentration = pd.DataFrame(results_concentration.squeeze().T)
            all_models_rate = pd.DataFrame(results_rate.squeeze().T)
            all_models = pd.concat([all_models_concentration, all_models_rate], axis=1)

            #
            # Save the Hyperparameters and the result of the single models of the ensemble
            #

            Model.save_pkl(self, all_models, "result_all_models",
                           os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))
            Model.save_csv(self, all_models, "result_all_models",
                           os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))
            Model.save_dict(self, {'Hyperparameters': self.hyperparameters_list}, "Hyperparameters",
                            os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))

            concentration0 = all_models_concentration.mean(axis=1).to_numpy()
            concentration1 = all_models_rate.mean(axis=1).to_numpy()
            testset_to_save["concentration0"] = concentration0
            testset_to_save["concentration1"] = concentration1

            hat_variable = str(self.forecast_variables[0]) + "_hat"

            testset_to_save[hat_variable] = concentration0

            mean = np.array(
                [tfp.distributions.Beta(concentration0=concentration0[x], concentration1=concentration1[x]).mean() for x
                 in
                 range(len(concentration0))])

            #
            # Get a point forecast
            #

            testset_to_save[hat_variable] = mean * (
                    self.modelparams_dict["scaleparam_max"] - self.modelparams_dict["scaleparam_min"]) + \
                                            self.modelparams_dict["scaleparam_min"]
            testset_to_save["scaleparam_min"] = self.modelparams_dict["scaleparam_min"]



        elif self.modelparams_dict["dist"] == "beta":

            results_concentration = np.array(
                [x(self.to_predict_features).parameters["distribution"].concentration0.numpy() for x in
                 self.localmodel])
            results_rate = np.array(
                [x(self.to_predict_features).parameters["distribution"].concentration1.numpy() for x in
                 self.localmodel])

            all_models_concentration = pd.DataFrame(results_concentration.squeeze().T)
            all_models_rate = pd.DataFrame(results_rate.squeeze().T)
            all_models = pd.concat([all_models_concentration, all_models_rate], axis=1)

            #
            # Save the Hyperparameters and the result of the single models of the ensemble
            #

            Model.save_pkl(self, all_models, "result_all_models",
                           os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))
            Model.save_csv(self, all_models, "result_all_models",
                           os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))
            Model.save_dict(self, {'Hyperparameters': self.hyperparameters_list}, "Hyperparameters",
                            os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))

            concentration0 = all_models_concentration.mean(axis=1).to_numpy()
            concentration1 = all_models_rate.mean(axis=1).to_numpy()
            testset_to_save["concentration0"] = concentration0
            testset_to_save["concentration1"] = concentration1

            hat_variable = str(self.forecast_variables[0]) + "_hat"

            testset_to_save[hat_variable] = concentration0

            mean = np.array(
                [tfp.distributions.Beta(concentration0=concentration0[x], concentration1=concentration1[x]).mean() for x
                 in
                 range(len(concentration0))])

            #
            # Get a point forecast
            #

            testset_to_save[hat_variable] = mean * self.modelparams_dict["scaleparam"]
            testset_to_save["scaleparam"] = self.modelparams_dict["scaleparam"]




        elif self.modelparams_dict["dist"] == "gamma":

            results_concentration = np.array(
                [x(self.to_predict_features).parameters["distribution"].concentration.numpy() for x in self.localmodel])
            results_rate = np.array(
                [x(self.to_predict_features).parameters["distribution"].rate.numpy() for x in self.localmodel])

            all_models_concentration = pd.DataFrame(results_concentration.squeeze().T)
            all_models_rate = pd.DataFrame(results_rate.squeeze().T)
            all_models = pd.concat([all_models_concentration, all_models_rate], axis=1)

            #
            # Save the Hyperparameters and the result of the single models of the ensemble
            #

            Model.save_pkl(self, all_models, "result_all_models",
                           os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))
            Model.save_csv(self, all_models, "result_all_models",
                           os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))
            Model.save_dict(self, {'Hyperparameters': self.hyperparameters_list}, "Hyperparameters",
                            os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))

            concentration = all_models_concentration.mean(axis=1).to_numpy()
            rate = all_models_rate.mean(axis=1).to_numpy()
            testset_to_save["concentration"] = concentration
            testset_to_save["rate"] = rate
            hat_variable = str(self.forecast_variables[0]) + "_hat"

            #
            # Get a point forecast through the mean of the distribution
            #

            mean = np.array([tfp.distributions.Gamma(concentration=concentration[x], rate=rate[x]).mean() for x in
                             range(len(concentration))])
            testset_to_save[hat_variable] = mean

        Model.save_csv(self, testset_to_save, self.name,
                       os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "testset_results"))
        Model.save_pkl(self, testset_to_save, self.name,
                       os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "testset_results"))
