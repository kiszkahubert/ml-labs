import keras.layers
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10, fashion_mnist, mnist
from keras.models import Model, Sequential
from keras.layers import Lambda,Flatten,Conv2DTranspose,GlobalAveragePooling2D,Dropout,Add,Concatenate,LeakyReLU,Activation,BatchNormalization,GlobalAveragePooling2D,Conv2D, Input, Dense, Reshape, Lambda, Average
from keras.optimizers import Adam
from keras.applications import xception, resnet, DenseNet201, VGG16
from keras import backend as K
from matplotlib import pyplot as plt

def zadanie12_1():
    (X_train,y_train),(X_test,y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = pd.get_dummies(pd.Categorical(y_train)).values
    y_test = pd.get_dummies(pd.Categorical(y_test)).values
    discriminator = Sequential(
        [
            Lambda(lambda x: K.expand_dims(x,axis=-1)),
            Conv2D(64,kernel_size=4,strides=2,padding='same'),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Conv2D(128,kernel_size=4,strides=2,padding='same'),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Conv2D(256, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Flatten(),
            Dense(1,activation='sigmoid')
        ],
        name="discriminator"
    )
    discriminator.compile(optimizer=Adam(0.001,beta_1=0.8),loss='binary_crossentropy',metrics=['accuracy'])
    # generator = Sequential(
    #     [
    #         Dense(256*4*4,activation='relu'),
    #         Reshape((4,4,256)),
    #         BatchNormalization(),
    #         Conv2DTranspose(256, kernel_size=4,strides=2,padding='same'),
    #         Activation('relu'),
    #         BatchNormalization(),
    #         Conv2DTranspose(128,kernel_size=4,strides=2,padding='same'),
    #         Activation('relu'),
    #         BatchNormalization(),
    #         Conv2DTranspose(64,kernel_size=4,strides=2,padding='same'),
    #         Activation('relu'),
    #         BatchNormalization(),
    #         Conv2DTranspose(3,kernel_size=4,strides=2,padding='same'),
    #         Activation('tanh')
    #     ],
    #     name="generator"
    # )
    # discriminator.compile(optimizer=Adam(0.0001),loss='binary_crossentropy',metrics=['accuracy'])
    # discriminator.trainable=False

if __name__ == '__main__':
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    zadanie12_1()