import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10, fashion_mnist, mnist
from keras.models import Model, Sequential
from keras.layers import Flatten,Add,Concatenate,Activation,BatchNormalization,GlobalAveragePooling2D,Conv2D, Input, Dense, Reshape, Lambda, Average
from keras.optimizers import Adam
from keras.applications import xception
from matplotlib import pyplot as plt

def zadanie11_1(X_train,X_test,y_train,y_test):
    def build_model(X_train,y_train,trainable):
        cls_cnt = np.unique(y_train).shape[0]
        input_tensor = Input(X_train.shape[1:])
        base_model = xception.Xception(include_top=False)
        output_tensor = Lambda(lambda x: xception.preprocess_input(x))(input_tensor)
        base_model.trainable = trainable
        output_tensor = base_model(output_tensor)
        output_tensor = Flatten()(output_tensor)
        output_tensor = Dense(cls_cnt,activation='softmax')(output_tensor)
        my_model = Model(inputs=input_tensor, outputs=output_tensor)
        my_model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
        return my_model
    
    def train_test(X_train,y_train,X_test,y_test):
        model = build_model(X_train,y_train,True)
        model.fit(X_train,y_train,epochs=10,batch_size=32)
        model = build_model(X_train,y_train,False)
        model.fit(X_train,y_train,epochs=10,batch_size=32)
        loss, acc = model.evaluate(X_test,y_test)
        print(f"Accuracy: {acc:.4f}")

    train_test(X_train,y_train,X_test,y_test)



if __name__ == '__main__':
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    (X_train,y_train),(X_test,y_test) = cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)
    zadanie11_1(X_train,X_test,y_train,y_test)