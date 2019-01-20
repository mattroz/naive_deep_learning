'''Implementation of various ResNet architectures (https://arxiv.org/abs/1512.03385)'''

import keras
from keras.layers import Input, BatchNormalization, Conv2D, Activation
from keras.models import Model


class StraightResNet18(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        """Build whole ResNet18 net"""

        '''Build initial block'''
        self.input = Input(shape=[self.input_shape[0], self.input_shape[1], 3], name='input')
        self.initial_conv = Conv2D(filters=64, kernel_size=7, strides=2, kernel_initializer='he_normal')(self.input)
        self.initial_bn = BatchNormalization()(self.initial_conv)
        self.initial_act = Activation('relu')(self.initial_bn)

