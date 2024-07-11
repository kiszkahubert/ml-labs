import numpy as np
from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from sklearn.model_selection import KFold

def build_model(num_layers, X_train, class_cnt):
    filter_cnt = 32
    learning_rate = 0.0001
    act_func = 'relu'
    kernel_size = (3,3)
    conv_rule = 'same'
    model = Sequential()
    model.add(Conv2D(input_shape=X_train.shape[1:],
                     filters=filter_cnt,
                     kernel_size=kernel_size,
                     padding=conv_rule,
                     activation=act_func))
    for _ in range(num_layers):
        model.add(Conv2D(filters=filter_cnt,
                         kernel_size=kernel_size,
                         padding=conv_rule,
                         activation=act_func))
    
    model.add(Flatten())
    model.add(Dense(class_cnt, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate),loss='SparseCategoricalCrossentropy',metrics=['accuracy'])
    return model

def evaluate_model():
    train, test = mnist.load_data()
    X_train, y_train = train[0], train[1]
    X_test, y_test = test[0], test[1]
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    class_cnt = np.unique(y_train).shape[0]
    accuracies = []
    for train_index, test_index in KFold(5).split(X_train):
        pass

def zadanie7_2():
    train, test = mnist.load_data()
    X_train, y_train = train[0], train[1]
    X_test, y_test = test[0], test[1]
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    class_cnt = np.unique(y_train).shape[0]
    filter_cnt = 32
    learning_rate = 0.0001
    act_func = 'relu'
    kernel_size = (3,3)
    model = Sequential()
    conv_rule = 'same'
    model.add(Conv2D(input_shape=X_train.shape[1:],
                     filters=filter_cnt,
                     kernel_size=kernel_size,
                     padding=conv_rule,
                     activation=act_func))
    model.add(Flatten())
    model.add(Dense(class_cnt,activation='softmax'))
    model.compile(optimizer=Adam(learning_rate),loss='SparseCategoricalCrossentropy',metrics=['accuracy'])
    model.fit(x=X_train, y=y_train, epochs=class_cnt, validation_data=(X_test,y_test))

if __name__ == "__main__":
    zadanie7_2()