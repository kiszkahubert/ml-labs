import tensorflow as tf
import pandas as pd
import numpy as np
import os
from keras.utils.vis_utils import plot_model
from keras.datasets import fashion_mnist, mnist
from keras.models import Model, Sequential, model_from_json
from keras.layers import Conv2D, MaxPool2D,GlobalAveragePooling2D,Dense,Input,Reshape,UpSampling2D,BatchNormalization,GaussianNoise
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot as plt


def test(X_train):
    act_func = "selu"
    aec_dim_num = 2
    encoder_layers = [
        GaussianNoise(1),
        BatchNormalization(),
        Conv2D(32,(7,7),padding='same',activation=act_func),
        MaxPool2D(2,2),
        BatchNormalization(),
        Conv2D(64,(5,5),padding='same',activation=act_func),
        MaxPool2D(2,2),
        BatchNormalization(),
        Conv2D(128,(3,3),padding='same',activation=act_func),
        GlobalAveragePooling2D(),
        Dense(aec_dim_num, activation='tanh')
    ]
    decoder_layers = [
        Dense(128,activation=act_func),
        BatchNormalization(),
        Reshape((1,1,128)),
        UpSampling2D((7,7)),
        Conv2D(32,(3,3),padding='same',activation=act_func),
        BatchNormalization(),
        UpSampling2D((2,2)),
        Conv2D(32,(5,5),padding='same',activation=act_func),
        BatchNormalization(),
        UpSampling2D((2,2)),
        Conv2D(32,(7,7),padding='same',activation=act_func),
        BatchNormalization(),
        Conv2D(1,(3,3),padding='same',activation='sigmoid')
    ]
    learning_rate = 0.0002
    tensor = input_aec = input_encoder = Input(X_train.shape[1:])
    for layer in encoder_layers:
        tensor = layer(tensor)
    
    output_encoder = tensor
    dec_tensor = input_decoder = Input(output_encoder.shape[1:])
    for layer in decoder_layers:
        tensor = layer(tensor)
        dec_tensor = layer(dec_tensor)
    
    output_aec = tensor
    output_decoder = dec_tensor
    autoencoder = Model(inputs=input_aec, outputs=output_aec)
    encoder = Model(inputs=input_encoder, outputs=output_encoder)
    decoder = Model(inputs=input_decoder, outputs=dec_tensor)
    autoencoder.compile(optimizer=Adam(learning_rate),loss='binary_crossentropy')
    autoencoder.fit(x=X_train,y=X_train,epochs=125,batch_size=256)
    save_dir = "./saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_json = autoencoder.to_json()
    with open(os.path.join(save_dir, "autoencoder.json"), "w") as json_file:
        json_file.write(model_json)

    autoencoder.save_weights(os.path.join(save_dir, "autoencoder_weights.h5"))
 
def load_model():
    with open('./saved_models/autoencoder.json','r') as json_file:
        loaded_model_json = json_file.read()
    
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('./saved_models/autoencoder_weights.h5')
    return loaded_model

if __name__ == '__main__':
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    (X_train,_), (_,_) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1) / 255.0
    load_model()