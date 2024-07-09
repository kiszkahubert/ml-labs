import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential # type: ignore
from keras.layers import Input, Dense # type: ignore
from keras.optimizers import Adam, SGD # type: ignore
from keras.utils import plot_model # type: ignore
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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

class zadanie5_3():
    def __init__(self) -> None:
        self.data = load_digits()
        self.X = self.data.data
        y = self.data.target
        self.class_num = len(np.unique(y))
        self.y = pd.Categorical(y)
        self.y = pd.get_dummies(self.y).values

    def create_model(self, num_layers=3, num_input=64, activation='relu', optimizer='adam', learning_rate=0.0001):
        model = Sequential()
        model.add(Dense(num_input, input_shape=(64,), activation=activation))
        for _ in range(num_layers - 1):
            model.add(Dense(num_input, activation=activation))
        
        model.add(Dense(self.class_num, activation='softmax'))
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate)

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def cross_validation(self, num_layers, num_input, activation, optimizer, learning_rate):
        accs = []
        scaler = StandardScaler()
        for train_index, test_index in KFold(5).split(self.X):
            X_train_cv, X_test_cv = self.X[train_index], self.X[test_index]
            y_train_cv, y_test_cv = self.y[train_index], self.y[test_index]
            X_train_cv = scaler.fit_transform(X_train_cv)
            X_test_cv = scaler.transform(X_test_cv)
            model = self.create_model(num_layers, num_input, activation, optimizer, learning_rate)
            model.fit(X_train_cv, y_train_cv, epochs=100, batch_size=32, verbose=0)
            y_pred_cv = np.argmax(model.predict(X_test_cv), axis=1)
            y_true_cv = np.argmax(y_test_cv, axis=1)
            acc = accuracy_score(y_true_cv, y_pred_cv)
            accs.append(acc)
    
        return np.mean(accs)
    
    def find_best_hyperparameters(self):
        num_layers_list = [2, 3, 4]
        num_input_list = [32, 64, 128]
        activation_list = ['relu', 'tanh']
        optimizer_list = ['adam', 'sgd']
        learning_rate_list = [0.001, 0.0001]
        best_acc = 0
        best_params = {}
        for num_layers in num_layers_list:
            for num_input in num_input_list:
                for activation in activation_list:
                    for optimizer in optimizer_list:
                        for learning_rate in learning_rate_list:
                            print(f'Testing: layers={num_layers}, inputs={num_input}, activation={activation}, optimizer={optimizer}, lr={learning_rate}')
                            acc = self.cross_validation(num_layers, num_input, activation, optimizer, learning_rate)
                            print(f'Accuracy: {acc}')
                            if acc > best_acc:
                                best_acc = acc
                                best_params = {
                                    'num_layers': num_layers,
                                    'num_input': num_input,
                                    'activation': activation,
                                    'optimizer': optimizer,
                                    'learning_rate': learning_rate
                                }

        print('Best Accuracy:', best_acc)
        print('Best Hyperparameters:', best_params)
        return best_acc, best_params


if __name__ == "__main__":
    zadanie = zadanie5_3()
    zadanie.find_best_hyperparameters() # Best Hyperparameters: {'num_layers': 2, 'num_input': 64, 'activation': 'relu', 'optimizer': 'adam', 'learning_rate': 0.001} 
