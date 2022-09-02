import tensorflow as tf
import tensorflow_probability as tfp
from forecasting_framework.model.distributions_self import IndependentGamma, IndependentBeta
from keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import elu
from tensorflow.python.keras import regularizers


class ProbMlp:
    feature_size = None
    dist = None

    def __init__(self, feature_size, dist):
        self.feature_size = feature_size
        self.dist = dist

    def create_model_inputs(self=None):

        return layers.Input(
            shape=(self.feature_size), dtype=tf.float32
        )

    def get_prob_model(self, hp):

        layer_1_size = hp.Int("first_layer", min_value=64, max_value=256, step=32)
        layer_2_size = hp.Int("second_layer", min_value=32, max_value=128, step=16)
        layer_3_size = hp.Int("third_layer", min_value=16, max_value=64, step=8)
        layer_4_size = hp.Int("fourth_layer", min_value=8, max_value=32, step=2)

        inputs = self.create_model_inputs()
        features = inputs

        features = Dense(layer_1_size, activation=elu, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.l2(1e-4), )(features)
        features = Dense(layer_2_size, activation=elu, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.l2(1e-4), )(features)
        features = Dense(layer_3_size, activation=elu, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.l2(1e-4), )(features)
        features = Dense(layer_4_size, activation=elu, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.l2(1e-4), )(features)
        fc = Dense(8)(features)

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

        model = keras.Model(inputs=inputs, outputs=outputs)

        return model
