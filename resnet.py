'''Implementation of various ResNet architectures (https://arxiv.org/abs/1512.03385)'''

import os
import keras
import tensorflow as tf
from keras.layers import Input, BatchNormalization, Conv2D, Activation, Dense, Flatten, Add, MaxPool2D
from keras.models import Model
import shutil


class StraightResNet18(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def shortcut(self, input, residual):
        input_shape = keras.backend.int_shape(input)
        residual_shape = keras.backend.int_shape(residual)
        stride_row = int(round(input_shape[1] / residual_shape[1]))
        stride_col = int(round(input_shape[2] / residual_shape[2]))
        channels_equal = (input_shape[3] == residual_shape[3])

        shortcut_connection = input
        '''If shapes are different, apply 1x1 conv'''
        if stride_col > 1 or stride_row > 1 or not channels_equal:
            shortcut_connection = Conv2D(filters=residual_shape[3], kernel_size=1, strides=(stride_row, stride_col),
                                         padding='valid', kernel_initializer='he_normal')(input)

        return Add()([shortcut_connection, residual])



    def residual_block(self, input, filters, block_idx, downsample=False):
        strides = 1
        if downsample:
            strides = 2

        bn1 = BatchNormalization(name=f'bn{block_idx}')(input)
        act1 = Activation('relu', name=f'act{block_idx}')(bn1)
        conv1 = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same',
                            kernel_initializer='he_normal', name=f'conv{block_idx}')(act1)

        bn2 = BatchNormalization(name=f'bn2_{block_idx}')(conv1)
        act2 = Activation('relu', name=f'act2_{block_idx}')(bn2)
        conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                              kernel_initializer='he_normal', name=f'conv2_{block_idx}')(act2)
        shortcut = self.shortcut(input, conv2)

        return shortcut


    def build(self):
        """Build whole ResNet18 net"""
        layer = []
        '''Build initial block'''
        filters_num = 64
        self.input = Input(shape=[self.input_shape[0], self.input_shape[1], 3], name='input')
        self.initial_conv = Conv2D(filters=filters_num, kernel_size=7, strides=2, padding='same',
                                   kernel_initializer='he_normal', name='initial_conv')(self.input)
        self.initial_bn = BatchNormalization(name='initial_BN')(self.initial_conv)
        self.initial_act = Activation('relu', name='initial_act')(self.initial_bn)
        self.initial_pool = MaxPool2D(pool_size=3, strides=2, padding='same')(self.initial_act)

        '''Start building net with basic blocks as in https://arxiv.org/pdf/1603.05027.pdf'''
        self.conv0 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer='he_normal', name='conv0')(self.initial_pool)

        layer.append(self.conv0)
        for i in range(8):
            layer.append(self.residual_block(input=layer[i],
                                        filters=filters_num,
                                        block_idx=(i + 1),
                                        downsample=((i+1) % 2 != 0)))
            filters_num *= 2 if ((i+1) % 2 == 0) else 1
            print(filters_num, ((i+1) % 2 == 0), i)

        self.flat = Flatten()(layer[-1])
        self.out = Dense(units=1000, name='FC1000')(self.flat)

        self.model = Model(inputs=self.input, outputs=self.out)


    def visualize_model(self):
        """Create tensorflow graph representation of ResNet model"""
        if os.path.isdir('./vis'):
            shutil.rmtree('./vis')
        os.mkdir(os.path.join(os.getcwd(), 'vis'))

        graph = tf.Graph()

        with graph.as_default():
            self.build()
            # compile method actually creates the model in the graph.
            self.model.compile(loss='categorical_crossentropy',
                               optimizer='adam', metrics=['accuracy'])
        writer = tf.summary.FileWriter(logdir='vis', graph=graph)
        writer.flush()

        return


def main():
    resnet = StraightResNet18((300,300))
    # resnet.build()
    resnet.visualize_model()

if __name__ == '__main__':
    main()