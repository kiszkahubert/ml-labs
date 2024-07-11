import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from keras.layers import Dense, BatchNormalization, Dropout, GaussianNoise, LayerNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2, l1
import tensorflow as tf

# Zadanie 6.1
def create_model(X, y, regularizer):
    neuron_num = 64
    learning_rate = 0.001
    model = Sequential()
    model.add(Dense(neuron_num, activation='relu', input_shape=(X.shape[1],)))
    repeat_num = 2
    for i in range(repeat_num):
        model.add(Dense(neuron_num, activation='selu', kernel_regularizer=regularizer))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
    return model

def cross_validation(X, y):
    regularizers = [l1(0.0), l1(0.0001), l1(0.001), l1(0.01), l1(0.1),
                    l2(0.0), l2(0.0001), l2(0.001), l2(0.01), l2(0.1)]
    scaler = StandardScaler()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for regularizer in regularizers:
        avg_accuracy_train = []
        avg_accuracy_val = []
        avg_loss_train = []
        avg_loss_val = []
        for train_index, test_index in KFold(5).split(X):
            X_train_cv, X_test_cv = X[train_index], X[test_index]
            y_train_cv, y_test_cv = y[train_index], y[test_index]
            X_train_cv = scaler.fit_transform(X_train_cv)
            X_test_cv = scaler.transform(X_test_cv)
            model = create_model(X, y, regularizer)
            history = model.fit(X_train_cv, y_train_cv, epochs=100, batch_size=32, validation_data=(X_test_cv, y_test_cv), verbose=0)
            avg_accuracy_train.append(history.history['accuracy'])
            avg_accuracy_val.append(history.history['val_accuracy'])
            avg_loss_train.append(history.history['loss'])
            avg_loss_val.append(history.history['val_loss'])
        
        avg_accuracy_train = np.mean(avg_accuracy_train, axis=0)
        avg_accuracy_val = np.mean(avg_accuracy_val, axis=0)
        avg_loss_train = np.mean(avg_loss_train, axis=0)
        avg_loss_val = np.mean(avg_loss_val, axis=0)
        axes[0].plot(avg_accuracy_train)
        axes[0].plot(avg_accuracy_val)
        axes[1].plot(avg_loss_train)
        axes[1].plot(avg_loss_val)
            
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    plt.show()

#Zadanie 6.2
def create_model_2(X, do_rate):
    neuron_num = 64
    learning_rate = 0.001
    block = [Dense,Dropout]
    args = [(neuron_num,'selu'),(do_rate,)]
    model = Sequential()
    model.add(Dense(neuron_num, activation='relu', input_shape=(X.shape[1],)))
    repeat_num = 2
    for i in range(repeat_num):
        for layer, arg in zip (block,args):
            model.add(layer(*arg))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
    return model    

def cross_validation_2(X, y):
    do_rates = [0,0.2,0.3,0.5]
    scaler = StandardScaler()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for rate in do_rates:
        avg_accuracy_train = []
        avg_accuracy_val = []
        avg_loss_train = []
        avg_loss_val = []
        for train_index, test_index in KFold(5).split(X):
            X_train_cv, X_test_cv = X[train_index], X[test_index]
            y_train_cv, y_test_cv = y[train_index], y[test_index]
            X_train_cv = scaler.fit_transform(X_train_cv)
            X_test_cv = scaler.transform(X_test_cv)
            model = create_model_2(X, rate)
            history = model.fit(X_train_cv, y_train_cv, epochs=100, batch_size=32, validation_data=(X_test_cv, y_test_cv), verbose=2)
            avg_accuracy_train.append(history.history['accuracy'])
            avg_accuracy_val.append(history.history['val_accuracy'])
            avg_loss_train.append(history.history['loss'])
            avg_loss_val.append(history.history['val_loss'])
        
        avg_accuracy_train = np.mean(avg_accuracy_train, axis=0)
        avg_accuracy_val = np.mean(avg_accuracy_val, axis=0)
        avg_loss_train = np.mean(avg_loss_train, axis=0)
        avg_loss_val = np.mean(avg_loss_val, axis=0)
        axes[0].plot(avg_accuracy_train)
        axes[0].plot(avg_accuracy_val)
        axes[1].plot(avg_loss_train)
        axes[1].plot(avg_loss_val)
            
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    plt.show()

#Zadanie 6.2
def create_model_3(X, noise):
    neuron_num = 64
    learning_rate = 0.001
    block = [Dense,GaussianNoise]
    args = [(neuron_num,'selu'),(noise,)]
    model = Sequential()
    model.add(Dense(neuron_num, activation='relu', input_shape=(X.shape[1],)))
    repeat_num = 2
    for i in range(repeat_num):
        for layer, arg in zip (block,args):
            model.add(layer(*arg))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
    return model    

def cross_validation_3(X, y):
    noises = [0,0.1,0.2,0.3]
    scaler = StandardScaler()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for noise in noises:
        avg_accuracy_train = []
        avg_accuracy_val = []
        avg_loss_train = []
        avg_loss_val = []
        for train_index, test_index in KFold(5).split(X):
            X_train_cv, X_test_cv = X[train_index], X[test_index]
            y_train_cv, y_test_cv = y[train_index], y[test_index]
            X_train_cv = scaler.fit_transform(X_train_cv)
            X_test_cv = scaler.transform(X_test_cv)
            model = create_model_3(X, noise)
            history = model.fit(X_train_cv, y_train_cv, epochs=100, batch_size=32, validation_data=(X_test_cv, y_test_cv), verbose=2)
            avg_accuracy_train.append(history.history['accuracy'])
            avg_accuracy_val.append(history.history['val_accuracy'])
            avg_loss_train.append(history.history['loss'])
            avg_loss_val.append(history.history['val_loss'])
        
        avg_accuracy_train = np.mean(avg_accuracy_train, axis=0)
        avg_accuracy_val = np.mean(avg_accuracy_val, axis=0)
        avg_loss_train = np.mean(avg_loss_train, axis=0)
        avg_loss_val = np.mean(avg_loss_val, axis=0)
        axes[0].plot(avg_accuracy_train)
        axes[0].plot(avg_accuracy_val)
        axes[1].plot(avg_loss_train)
        axes[1].plot(avg_loss_val)
            
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    plt.show()



if __name__ == "__main__":
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    data = load_breast_cancer()
    X = data.data
    y = data.target
    cross_validation_2(X, y)
