import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Beta
from tensorflow_probability.python.distributions import distribution as tfd
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions.gamma import Gamma
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.layers.distribution_layer import DistributionLambda, _event_size, _serialize, \
    _get_convert_to_tensor_fn


#
#
# This source code with it's comments is nearly analogue to the sourcecode
# of independent normal, of the package tfp, only for the beta and gamma distribution
#
#
#


class IndependentGamma(DistributionLambda):
    """An independent beta Keras layer.
    ### Analogue to the `IndependentNormal` layer of tensorflow
    """

    def __init__(self,
                 event_shape=(),
                 convert_to_tensor_fn=tfd.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        """Initialize the `IndependentGamma` layer.
        Args:
          event_shape: integer vector `Tensor` representing the shape of single
            draw from this distribution.
          convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
            instance and returns a `tf.Tensor`-like object.
            Default value: `tfd.Distribution.sample`.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
            Default value: `False`.
          **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
        """
        convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

        # If there is a 'make_distribution_fn' keyword argument (e.g., because we
        # are being called from a `from_config` method), remove it.  We pass the
        # distribution function to `DistributionLambda.__init__` below as the first
        # positional argument.
        kwargs.pop('make_distribution_fn', None)

        super(IndependentGamma, self).__init__(
            lambda t: IndependentGamma.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs)

        self._event_shape = event_shape
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector.
         """
        with tf.name_scope(name or 'IndependentGamma'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = dist_util.expand_to_vector(
                tf.convert_to_tensor(
                    event_shape, name='event_shape', dtype_hint=tf.int32),
                tensor_name='event_shape')
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape,
            ],
                axis=0)
            loc_params, scale_params = tf.split(params, 2, axis=-1)
            return independent_lib.Independent(
                Gamma(
                    concentration=tf.reshape(loc_params, output_shape),
                    rate=tf.reshape(scale_params, output_shape),
                    validate_args=validate_args),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution.
        """
        with tf.name_scope(name or 'IndependentGamma_params_size'):
            event_shape = tf.convert_to_tensor(
                event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(2) * _event_size(
                event_shape, name=name or 'IndependentGamma_params_size')

    def get_config(self):
        """Returns the config of this layer.
        NOTE: At the moment, this configuration can only be serialized if the
        Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
        implements `get_config`) or one of the standard values.

        """
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
            'validate_args': self._validate_args
        }
        base_config = super(IndependentGamma, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IndependentBeta(DistributionLambda):
    """An independent beta Keras layer.
    ### Analogue to the `IndependentNormal` layer of tensorflow

    """

    def __init__(self,
                 event_shape=(),
                 convert_to_tensor_fn=tfd.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        """Initialize the `IndependentBeta` layer.
        Args:
          event_shape: integer vector `Tensor` representing the shape of single
            draw from this distribution.
          convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
            instance and returns a `tf.Tensor`-like object.
            Default value: `tfd.Distribution.sample`.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
            Default value: `False`.
          **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
        """
        convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

        # If there is a 'make_distribution_fn' keyword argument (e.g., because we
        # are being called from a `from_config` method), remove it.  We pass the
        # distribution function to `DistributionLambda.__init__` below as the first
        # positional argument.
        kwargs.pop('make_distribution_fn', None)

        super(IndependentBeta, self).__init__(
            lambda t: IndependentBeta.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs)

        self._event_shape = event_shape
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector.

        """
        with tf.name_scope(name or 'IndependentBeta'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = dist_util.expand_to_vector(
                tf.convert_to_tensor(
                    event_shape, name='event_shape', dtype_hint=tf.int32),
                tensor_name='event_shape')
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape,
            ],
                axis=0)
            con0_params, con1_params = tf.split(params, 2, axis=-1)
            return independent_lib.Independent(
                Beta(
                    concentration0=tf.reshape(con0_params, output_shape),
                    concentration1=tf.reshape(con1_params, output_shape),
                    validate_args=validate_args),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution.
        """
        with tf.name_scope(name or 'IndependentBeta_params_size'):
            event_shape = tf.convert_to_tensor(
                event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(2) * _event_size(
                event_shape, name=name or 'IndependentBeta_params_size')

    def get_config(self):
        """Returns the config of this layer.
        NOTE: At the moment, this configuration can only be serialized if the
        Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
        implements `get_config`) or one of the standard values.

        """
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
            'validate_args': self._validate_args
        }
        base_config = super(IndependentBeta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
