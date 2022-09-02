import tensorflow as tf
import tensorflow_probability as tfp
from forecasting_framework.model.distributions_self import IndependentGamma, IndependentBeta
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.activations import elu, linear
from tensorflow.keras.layers import Conv1D, Dense, Input, Reshape, MaxPool1D, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import BatchNormalization


class ProbCNN:
    feature_size = None
    lag_size = None
    dist = None

    def __init__(self, feature_size, lag_size, dist):
        self.feature_size = feature_size
        self.lag_size = lag_size
        self.dist = dist

    def get_prob_cnn_model(self, hp):

        #
        # Feature network
        #

        feature_input = Input(shape=(24, self.feature_size), name="feature_input")

        #
        # First conv layer feature
        #

        feature = Conv1D(hp.Int("filter_size_feature_l1", min_value=1, max_value=300),
                         [hp.Int("window_size_feature_l1", min_value=1, max_value=24)],
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.l2(1e-4),
                         activation=linear, padding='same')(feature_input)
        feature = BatchNormalization()(feature)
        feature = elu(feature)
        feature = MaxPool1D(pool_size=2)(feature)

        #
        # Second conv layer feature
        #

        feature = Conv1D(hp.Int("filter_size_feature_l2", min_value=1, max_value=20),
                         [hp.Int("window_size_feature_l2", min_value=1, max_value=4)],
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.l2(1e-4),
                         activation=linear, padding='same')(feature)
        feature = BatchNormalization()(feature)
        feature = elu(feature)
        feature = MaxPool1D(pool_size=2)(feature)

        #
        # Dense network last layer feature
        #

        feature = Flatten()(feature)

        feature = Dense(48,
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activation=linear)(feature)
        feature = BatchNormalization()(feature)
        feature = elu(feature)

        #
        # Lag network part
        #

        lag_input = Input(shape=(24, self.lag_size), name="lag_input")

        #
        # First layer lag network
        #

        lag = Conv1D(hp.Int("filter_size_lag_l1", min_value=1, max_value=300),
                     [hp.Int("window_size_lag_l1", min_value=1, max_value=24)],
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activation=linear, padding='same')(lag_input)
        lag = BatchNormalization()(lag)
        lag = elu(lag)
        lag = MaxPool1D(pool_size=2)(lag)

        #
        # Second layer lag network
        #

        lag = Conv1D(hp.Int("filter_size_lag_l2", min_value=1, max_value=20),
                     [hp.Int("window_size_lag_l2", min_value=1, max_value=4)],
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activation=linear, padding='same')(lag)
        lag = BatchNormalization()(lag)
        lag = elu(lag)
        lag = MaxPool1D(pool_size=2)(lag)

        #
        # Dense network and last layer of lag
        #

        lag = Flatten()(lag)
        lag = Dense(48,
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activation=linear)(lag)
        lag = BatchNormalization()(lag)
        lag = elu(lag)

        #
        # Reshape to get the right sizes
        #

        feature = Reshape((48, 1))(feature)
        lag = Reshape((48, 1))(lag)

        #
        # Fully connected layer at the end
        #

        fc = concatenate([feature, lag], axis=2)

        fc = Flatten()(fc)

        fc = Dense(64, activation=linear, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                   bias_regularizer=regularizers.l2(1e-4), )(fc)
        fc = BatchNormalization()(fc)
        fc = elu(fc)

        fc = Dense(32, activation=linear, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                   bias_regularizer=regularizers.l2(1e-4), )(fc)
        fc = BatchNormalization()(fc)
        fc = elu(fc)

        fc = Dense(8, activation=linear, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                   bias_regularizer=regularizers.l2(1e-4), )(fc)
        fc = BatchNormalization()(fc)
        fc = elu(fc)

        #
        # End layer distribution
        #

        if self.dist == "normal":
            distribution_params = layers.Dense(units=2)(fc)
            outputs = tfp.layers.IndependentNormal(1)(distribution_params)

        elif self.dist == "gamma":
            distribution_params = layers.Dense(units=2, activation=tf.math.softplus)(fc)
            outputs = IndependentGamma(1)(distribution_params)

        elif self.dist == "beta":
            distribution_params = layers.Dense(units=2, activation=tf.math.softplus)(fc)
            outputs = IndependentBeta(1)(distribution_params)

        elif self.dist == "beta_with_min_max":
            distribution_params = layers.Dense(units=2, activation=tf.math.softplus)(fc)
            outputs = IndependentBeta(1)(distribution_params)
        else:
            distribution_params = layers.Dense(units=2)(fc)
            outputs = tfp.layers.IndependentNormal(1)(distribution_params)

        model = Model(inputs=[feature_input, lag_input], outputs=outputs)

        return model
