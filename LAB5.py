import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential # type: ignore
from keras.layers import Input, Dense # type: ignore
from keras.optimizers import Adam, RMSprop, SGD # type: ignore
from keras.utils import plot_model # type: ignore

def zadanie5_2():
    data = load_digits()
    X = data.data
    y = data.target
    class_num = len(np.unique(y))
    y = pd.Categorical(y)
    y = pd.get_dummies(y).values
    model = Sequential()
    model.add(Dense(64, input_shape = (X.shape[1],),activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(class_num, activation = 'softmax'))
    learning_rate = 0.0001
    model.compile(optimizer = Adam(learning_rate), loss='categorical_crossentropy',metrics=['accuracy'])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_test,y_test),verbose=2)
    model.summary()

def zadanie5_3():
    pass

if __name__ == "__main__":
    zadanie5_2()
