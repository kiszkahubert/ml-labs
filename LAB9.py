import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.datasets import fashion_mnist, mnist
from keras.models import Model, Sequential
from keras.layers import Add,Concatenate,Activation,BatchNormalization,GlobalAveragePooling2D,Conv2D, Input, Dense, Reshape, Lambda, Average
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot as plt
from matplotlib import rcParams
from PIL import Image

#zadanie 9.1
def build_model(X_train,X_test,y_train,y_test):
    output_tensor = input_tensor = Input(X_train.shape[1:])
    act_func = 'relu'
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
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=X_train,y=y_train,epochs=10,batch_size=32, validation_data=(X_test,y_test))
    plot_model(model,show_shapes=True)

#zadanie 9.2
def resnet_block(input_tensor):
    output_tensor = Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu')(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Add()([output_tensor,input_tensor])
    output_tensor = Activation('relu')(output_tensor)
    return output_tensor

def build_res_net(count, X_train):
    output_tensor = input_tensor = Input(X_train.shape[1:])
    for _ in range(count):
        output_tensor = resnet_block(output_tensor)
        
    model = Model(inputs=input_tensor,outputs=output_tensor)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    plot_model(model,show_shapes=True)

#zadanie 9.3
def densenet_block(input_tensor):
    output_tensor = Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu')(input_tensor)        
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Concatenate()([output_tensor,input_tensor])
    output_tensor = Conv2D(filters=32,kernel_size=(1,1),padding='same',activation='relu')(output_tensor)
    return output_tensor

def build_densenet(count, X_train):
    output_tensor = input_tensor = Input(X_train.shape[1:])
    for _ in range(count):
        output_tensor = resnet_block(output_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    plot_model(model,show_shapes=True)

#zadanie 9.4
def MISH(tensor):
    return tensor*tf.math.tanh(tf.math.log(1+tf.math.exp(tensor)))

def build_nnetwork(count, X_train,X_test,y_train,y_test):
    output_tensor = input_tensor = Input(X_train.shape[1:])
    for _ in range(count):
        output_tensor = densenet_block(output_tensor)
        output_tensor = Lambda(MISH)(output_tensor)

    output_tensor = GlobalAveragePooling2D()(output_tensor)
    output_tensor = Dense(10, activation='softmax')(output_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=X_train,y=y_train,epochs=10,batch_size=32,validation_data=(X_test,y_test))

if __name__ == '__main__':
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    (X_train,y_train), (X_test,y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = pd.get_dummies(pd.Categorical(y_train)).values
    y_test = pd.get_dummies(pd.Categorical(y_test)).values
    # build_model(X_train,X_test,y_train,y_test)
    # build_res_net(3,X_train)
    build_nnetwork(2,X_train,X_test,y_train,y_test)
