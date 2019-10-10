import keras
import math
from keras import backend as K
from keras.layers import Layer


class TuningActivationLayer(Layer):

    def __init__(self, activationFunction, **kwargs):
        self._af = activationFunction
        super(TuningActivationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._xs = self.add_weight(name='horizontalScale', shape=(1,), initializer=keras.initializers.Constant(value=1), trainable=True)
        self._xb = self.add_weight(name='horizontalBias',  shape=(1,), initializer=keras.initializers.Constant(value=0), trainable=True)
        self._ys = self.add_weight(name='verticalScale',   shape=(1,), initializer=keras.initializers.Constant(value=1), trainable=True)
        self._yb = self.add_weight(name='verticalBias',    shape=(1,), initializer=keras.initializers.Constant(value=0), trainable=True)
        super(TuningActivationLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        return self._ys * self._af((self._xs * x) + self._xb) + self._yb

    def compute_output_shape(self, input_shape):
        return input_shape


class LearnedFourierActivationLayer(Layer):

    def __init__(self, paramCount, **kwargs):
        self._N = paramCount
        super(LearnedFourierActivationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        self._p = self.add_weight(name='p', shape=(self._N,), initializer=keras.initializers.Constant(value=1), trainable=True)

        self._q = self.add_weight(name='q', shape=(self._N,), initializer=keras.initializers.Constant(value=1), trainable=True)

        super(LearnedFourierActivationLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):

        s = 0
        for i in range(0, self._N):
            s = s + (self._p[i] * K.cos(2 * math.pi * i * x)) + (self._q[i] * K.sin(2 * math.pi * i * x))
        return s

    def compute_output_shape(self, input_shape):
        return input_shape


class LearnedPolynomialActivationLayer(Layer):

    def __init__(self, paramCount, **kwargs):
        self._N = paramCount
        super(LearnedPolynomialActivationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        self._p = self.add_weight(name='p', shape=(self._N,), initializer=keras.initializers.RandomNormal(mean=1, stddev=0.05), trainable=True)

        self._q = self.add_weight(name='q', shape=(self._N,), initializer=keras.initializers.RandomNormal(mean=1, stddev=0.05), trainable=True)

        super(LearnedPolynomialActivationLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):

        numerator = 0
        denominator = 0
        for i in range(0, self._N):
            numerator += self._p[i] * K.pow(x, i)
            denominator += self._q[i] * K.pow(x, i)
        return numerator/denominator

    def compute_output_shape(self, input_shape):
        return input_shape

class LearnedExponentialActivationLayer(Layer):

    def __init__(self, paramCount, **kwargs):
        self._N = paramCount
        super(LearnedExponentialActivationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        self._p = self.add_weight(name='p', shape=(self._N,), initializer=keras.initializers.RandomNormal(mean=0, stddev=0.005), trainable=True)
        self._q = self.add_weight(name='q', shape=(self._N,), initializer=keras.initializers.RandomNormal(mean=0, stddev=0.005), trainable=True)
        super(LearnedExponentialActivationLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        s = 0
        for i in range(0, self._N):
            s += self._p[i]*K.exp(i * self._q[i] * x)

        return s

    def compute_output_shape(self, input_shape):
        return input_shape
