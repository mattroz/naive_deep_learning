'''Implementation of various ResNet architectures (https://arxiv.org/abs/1512.03385)'''

import keras
import tensorflow as tf
from keras.layers import Input, BatchNormalization, Conv2D, Activation, Dense, Flatten
from keras.models import Model


class StraightResNet18(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def build(self):
        """Build whole ResNet18 net"""

        '''Build initial block'''
        self.input = Input(shape=[self.input_shape[0], self.input_shape[1], 3], name='input')
        self.initial_conv = Conv2D(filters=64, kernel_size=7, strides=2,
                                   kernel_initializer='he_normal', name='initial_conv')(self.input)
        self.initial_bn = BatchNormalization(name='initial_BN')(self.initial_conv)
        self.initial_act = Activation('relu', name='initial_act')(self.initial_bn)

        '''Start building net with basic blocks as in https://arxiv.org/pdf/1603.05027.pdf'''
        self.conv0 = Conv2D(filters=64, kernel_size=3, strides=1,
                            kernel_initializer='he_normal', name='conv0')(self.initial_act)

        '''Block 1'''
        self.bn1 = BatchNormalization(name='bn1')(self.conv0)
        self.act1 = Activation('relu', name='act1')(self.bn1)
        self.conv1 = Conv2D(filters=64, kernel_size=3, strides=1,
                            kernel_initializer='he_normal', name='conv1')(self.act1)

        '''Block 2'''
        self.bn2 = BatchNormalization(name='bn2')(self.conv1)
        self.act2 = Activation('relu', name='act2')(self.bn2)
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=1,
                            kernel_initializer='he_normal', name='conv2')(self.act2)

        '''Block 3'''
        self.bn3 = BatchNormalization(name='bn3')(self.conv2)
        self.act3 = Activation('relu', name='act3')(self.bn3)
        self.conv3 = Conv2D(filters=128, kernel_size=3, strides=2,
                            kernel_initializer='he_normal', name='conv3')(self.act3)

        '''Block 4'''
        self.bn4 = BatchNormalization(name='bn4')(self.conv3)
        self.act4 = Activation('relu', name='act4')(self.bn4)
        self.conv4 = Conv2D(filters=128, kernel_size=3, strides=1,
                            kernel_initializer='he_normal', name='conv4')(self.act4)

        '''Block 5'''
        self.bn5 = BatchNormalization(name='bn5')(self.conv4)
        self.act5 = Activation('relu', name='act5')(self.bn5)
        self.conv5 = Conv2D(filters=256, kernel_size=3, strides=2,
                            kernel_initializer='he_normal', name='conv5')(self.act5)

        '''Block 6'''
        self.bn6 = BatchNormalization(name='bn6')(self.conv5)
        self.act6 = Activation('relu', name='act6')(self.bn6)
        self.conv6 = Conv2D(filters=256, kernel_size=3, strides=1,
                            kernel_initializer='he_normal', name='conv6')(self.act6)

        '''Block 7'''
        self.bn7 = BatchNormalization(name='bn7')(self.conv6)
        self.act7 = Activation('relu', name='act7')(self.bn7)
        self.conv7 = Conv2D(filters=512, kernel_size=3, strides=2,
                            kernel_initializer='he_normal', name='conv7')(self.act7)

        '''Block 8'''
        self.bn8 = BatchNormalization(name='bn8')(self.conv7)
        self.act8 = Activation('relu', name='act8')(self.bn8)
        self.conv8 = Conv2D(filters=512, kernel_size=3, strides=1,
                            kernel_initializer='he_normal', name='conv8')(self.act8)
        self.flat = Flatten()(self.conv8)
        self.out = Dense(units=1000, name='FC1000')(self.flat)

        self.model = Model(inputs=self.input, outputs=self.out)


    def visualize_model(self):
        """Create tensorflow graph representation of ResNet model"""

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
    resnet.build()
    resnet.visualize_model()

if __name__ == '__main__':
    main()