from tensorflow.keras import regularizers
from tensorflow.keras.activations import elu, linear
from tensorflow.keras.layers import Conv1D, Dense, Input, Reshape, MaxPool1D, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import BatchNormalization


class CNN:
    feature_size = None
    lag_size = None

    def __init__(self, feature_size, lag_size):
        self.feature_size = feature_size
        self.lag_size = lag_size

    # body of the constructor

    def get_cnn(self, hp):
        """
        get an instance of the cnn network
        :param n_steps_out: The number of prediction, the network should make (horizon)
        :return: keras model

        """

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
        # Dense network
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

        lag = Flatten()(lag)

        #
        # Dense network and last layer of lag
        #

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

        fc = Dense(1)(fc)
        pred = fc

        model = Model(inputs=[feature_input, lag_input], outputs=pred)

        return model
