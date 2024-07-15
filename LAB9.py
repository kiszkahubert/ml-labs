import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.datasets import fashion_mnist, mnist
from keras.models import Model, Sequential
from keras.layers import BatchNormalization,GlobalAveragePooling2D,Conv2D, Input, Dense, Reshape, Lambda, Average
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot as plt
from matplotlib import rcParams
from PIL import Image

#zadanie 9.1
def build_model():
    (X_train,y_train), (X_test,y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = pd.get_dummies(pd.Categorical(y_train)).values
    y_test = pd.get_dummies(pd.Categorical(y_test)).values
    output_tensor = input_tensor = Input(X_train.shape[1:])
    filter_cnt = 32
    kernel_size = (3,3)
    act_func = 'selu'
    class_cnt = y_train.shape[1]
    output_tensor = Reshape((784,))(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)

    branch_1 = Dense(512, activation=act_func)(output_tensor)
    branch_1 = Dense(128, activation=act_func)(branch_1)
    branch_1 = Dense(64, activation=act_func)(branch_1)
    branch_1 = Dense(16, activation=act_func)(branch_1)
    branch_1 = Dense(10, activation=act_func)(branch_1)
    branch_2 = Dense(512, activation=act_func)(output_tensor)
    branch_2 = Dense(64, activation=act_func)(branch_2)
    branch_2 = Dense(10, activation=act_func)(branch_2)
    branch_3 = Dense(512, activation=act_func)(output_tensor)
    branch_3 = Dense(64, activation=act_func)(branch_3)
    branch_3 = Dense(10, activation=act_func)(branch_3)
    branch_4 = Dense(512, activation=act_func)(output_tensor)
    branch_4 = Dense(64, activation=act_func)(branch_4)
    branch_4 = Dense(10, activation=act_func)(branch_4)
    branch_5 = Dense(512, activation=act_func)(output_tensor)
    branch_5 = Dense(64, activation=act_func)(branch_5)
    branch_5 = Dense(10, activation=act_func)(branch_5)
    merged = Average()([branch_1,branch_2,branch_3,branch_4,branch_5])
    model = Model(inputs=input_tensor,outputs=merged)
    md = model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    plot_model(md,show_shapes=True)

if __name__ == '__main__':
    build_model()