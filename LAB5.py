import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential # type: ignore
from keras.layers import Input, Dense # type: ignore
from keras.optimizers import Adam, RMSprop, SGD # type: ignore
from keras.utils import plot_model # type: ignore

if __name__ == "__main__":
    data = load_iris()
    y = data.target
    X = data.data
    y = pd.Categorical(y)
    y = pd.get_dummies(y).values
    class_num = y.shape[1]
    model = Sequential()
    model.add(Dense(64, input_shape = (X.shape[1],),activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(class_num, activation = 'softmax'))
    learning_rate = 0.0001
    model.compile(optimizer = Adam(learning_rate), loss='categorical_crossentropy',metrics=('accuracy',))
    model.summary()
    plot_model(model, to_file="my_model.png")
