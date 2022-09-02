import random as python_random
import sys

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
)
from sklearn.preprocessing import StandardScaler

sys.path.append(".")

from forecasting_framework.model.model_cnn import CNN

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

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


import os
import math
from forecasting_framework.model.model import Model
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import model_selection


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
        for train_indices, test_indices in cv.split(x[0]):
            x_train, x_test = [x[0][train_indices], x[1][train_indices]], [x[0][test_indices],
                                                                           x[1][test_indices]]
            y_train, y_test = y[train_indices], y[test_indices]
            tf.keras.backend.clear_session()

            model = self.hypermodel.build(trial.hyperparameters)
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
            print("CV " + str(count) + " out of " + str(folds) + " started.")
            model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test),
                      callbacks=[
                          tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                                           patience=self.bo_patience, min_delta=0.001,
                                                           restore_best_weights=True)],
                      epochs=self.bo_epochs, verbose=1, shuffle=True)

            val_losses.append(model.evaluate(x_test, y_test))
            print("CV " + str(count) + " out of " + str(folds) + " ended.")

            print("Val_losses : " + str(val_losses))
            count = count + 1

        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
        self.save_model(trial.trial_id, model)


class BoCNN(Model):
    """
    Performs bayesian optimisation for the network in the training method, then applies the model in the test method.
    """

    data = None
    to_predict_features = None
    scaler_forecast_varaibles = None
    scaler_features = None
    scaler_lag = None
    localmodel = None
    hyperparameters_list = None
    train_epochs = 1000
    train_patience = 15

    def __init__(self, data_path,
                 modelparams_dict, features, forecast_variables, modelname, train, folder, name, plot_diagnostic=False,
                 save_diagnostics=False):
        Model.__init__(self, data_path=data_path, modelparams=modelparams_dict,
                       features=features,
                       forecast_variables=forecast_variables, folder=folder, plot_diagnostic=plot_diagnostic,
                       train=train,
                       save_diagnostics=save_diagnostics, name=name, modelname="nn")

    def execute_train(self):

        #
        #   Prevent double running of the models
        #

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

        # Seperate lag features

        list_lag_features = [x for x in self.features if "lag_" in x]
        other_features = [x for x in self.features if not "lag_" in x]

        # fit scaler on val and train data

        train_val = x_with_lag[x_with_lag.index.year <= 2018]

        self.scaler_lag = StandardScaler().fit(train_val[list_lag_features])
        self.scaler_features = StandardScaler().fit(train_val[other_features])
        self.scaler_forecast_varaibles = StandardScaler().fit((y[y.index.year <= 2018]))

        y = self.scaler_forecast_varaibles.transform(y)
        lag = self.scaler_lag.transform(x_with_lag[list_lag_features])
        x = self.scaler_features.transform(x_with_lag[other_features])

        # Input for noise model
        x_split = []
        # ground truth
        y_split_res = []
        # Input for lag model
        lag_split = []

        for i in range(x.shape[0] - 48):
            y_split_res.append(y[48 + i])
            x_split.append(x[25 + i:49 + i])
            lag_split.append(lag[25 + i:49 + i])

        print("Done")

        #
        # Split into train test val
        #

        end_val = len(train_val) - 48
        x_split = np.array(x_split)
        y_split_res = np.array(y_split_res)
        lag_split = np.array(lag_split)

        x_train_val = x_split[0:end_val]
        y_train_val = y_split_res[0:end_val]
        lag_split_train_val = lag_split[0:end_val]

        x_test = x_split[end_val:]
        y_test = y_split_res[end_val:]
        lag_split_test = lag_split[end_val:]

        print("Shape Test/Val Size :")
        print(x_train_val.shape)
        print(y_train_val.shape)

        print("Shape Test Size :")
        print(x_test.shape)
        print(y_test.shape)

        features_train_val = [x_train_val, lag_split_train_val]
        features_test = [x_test, lag_split_test]

        cnn = CNN(feature_size=features_train_val[0].shape[2], lag_size=features_train_val[1].shape[2])

        # Fixes working directory issues on bw uni cluster
        working_dir = os.getcwd()
        if working_dir.split("/")[-1] != "forecasting_framework":
            working_dir = working_dir + "/forecasting_framework"

        tuner = CVTuner(
            hypermodel=cnn.get_cnn,
            oracle=kt.oracles.BayesianOptimization(
                objective=kt.Objective('val_loss', direction="min"),
                max_trials=100, seed=42),
            directory=str(working_dir) + "/models/baysian_results/" + str(self.forecast_variables[0]),
            project_name=str(self.folder),
            logger=TensorBoardLogger(
                metrics=['val_loss', 'loss'],
                logdir=str(working_dir) + "/models/baysian_results/" + str(self.forecast_variables[0]) + "/" + str(
                    self.folder) + "_tb" + "/logs/hparams")
        )

        tuner.search(features_train_val, y_train_val)

        tuner.search_space_summary()

        print("Done")

        self.to_predict_features = features_test

        best_hp = tuner.get_best_hyperparameters()[0]
        tuner.results_summary()
        self.hyperparameters_list = [x.get_config() for x in tuner.get_best_hyperparameters()]

        # TRAIN the real models


        self.localmodel = []

        folds = 4
        cv = model_selection.TimeSeriesSplit(n_splits=folds, test_size=8760)
        count = 1

        for train_indices, test_indices in cv.split(features_train_val[0]):
            # Seed everything again bevor training
            np.random.seed(42)
            python_random.seed(42)
            tf.compat.v1.set_random_seed(42)

            tmp_model = cnn.get_cnn(best_hp)
            tmp_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

            x_train_fold, x_test_fold = [features_train_val[0][train_indices], features_train_val[1][train_indices]], [
                features_train_val[0][test_indices], features_train_val[1][test_indices]]

            y_train_fold, y_test_fold = y_train_val[train_indices], y_train_val[test_indices]
            print("Real training of model " + str(count) + " out of " + str(folds) + " began.")
            history = tmp_model.fit(x_train_fold, y_train_fold, batch_size=128, validation_data=(x_test_fold, y_test_fold),
                                    callbacks=[
                                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                                                         patience=self.train_patience,
                                                                         restore_best_weights=True, min_delta=0.001)
                                    ],
                                    epochs=self.train_epochs, verbose=1, shuffle=True)
            self.localmodel.append(tmp_model)

            print("Real Training of Model " + str(count) + " out of " + str(folds) + " ended.")
            count = count + 1

    #
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

        #
        # Get the predictions
        #

        results_of_all_models = np.array(
            [self.scaler_forecast_varaibles.inverse_transform(x.predict(self.to_predict_features)) for x in
             self.localmodel])
        all_models = pd.DataFrame(results_of_all_models.squeeze().T)

        #
        # Save the results and Hyperparameters
        #

        Model.save_pkl(self, all_models, "result_all_models",
                       os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))
        Model.save_csv(self, all_models, "result_all_models",
                       os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))
        Model.save_dict(self, {'Hyperparameters': self.hyperparameters_list}, "Hyperparameters",
                        os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "all_models"))

        #
        # Ensemble
        #

        y_hat = results_of_all_models.mean(axis=0)

        testset_to_save = pd.read_pickle(testset_path)[-8760:]
        testset_to_save[str(self.forecast_variables[0]) + "_hat"] = y_hat[:]

        variable = str(self.forecast_variables[0])
        hat_variable = str(self.forecast_variables[0]) + "_hat"
        result_df_test = testset_to_save

        rmse_models = []
        for x in range(results_of_all_models.shape[0]):
            res_single_model = results_of_all_models[x, :].squeeze()
            rmse_single_model = math.sqrt(((result_df_test[variable] - res_single_model) ** 2).mean())
            rmse_models.append(rmse_single_model)

        rmse = math.sqrt(((result_df_test[variable] - result_df_test[hat_variable]) ** 2).mean())
        print("RMSE TEST Models " + str(rmse_models))
        print("RMSE TEST " + str(rmse))

        #
        # Save Final Result
        #

        Model.save_csv(self, testset_to_save, self.name,
                       os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "testset_results"))
        Model.save_pkl(self, testset_to_save, self.name,
                       os.path.join(str(self.forecast_variables[0]) + "/" + self.folder, "testset_results"))
