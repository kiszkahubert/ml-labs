import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Input, LSTM, Dense, TimeDistributed
from keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score

def make_dataset(dataset,obs_length,target_feature):
    total_ds_length = len(dataset)
    X_length = total_ds_length-obs_length
    X = np.zeros((X_length,obs_length,dataset.shape[1]))
    for i in range(X_length):
        X[i,...] = dataset[i:i+obs_length,...]
    
    y = dataset[obs_length:, target_feature]
    return X,y

def normalize_array(train,test,axis):
    means = train.mean(axis=axis)
    stds = train.mean(axis=axis)
    train = (train - means)/stds
    test = (test - means)/stds
    return train, test

def return_selected():
    data = pd.read_csv('files/lab 13/prices.csv')
    descr = pd.read_csv('files/lab 13/securities.csv')
    selected_data = data[data['symbol']=='YHOO']
    selected_data = selected_data.sort_values('date')
    selected_data = selected_data.drop(columns=['date','symbol'])
    selected_data = selected_data.values
    return selected_data

def zadanie1():
    selected_data = return_selected()
    X,y = make_dataset(selected_data,100,-3)
    train_size = int(X.shape[0]*0.8)
    X_train,y_train = X[:train_size,...],y[:train_size,...]
    X_test,y_test = X[train_size:,...],y[train_size:,...]
    X_train, X_test = normalize_array(X_train,X_test,0)
    y_train, y_test = normalize_array(y_train,y_test,0)
    input_tensor = output_tensor = Input(X_train.shape[1:])
    output_tensor = TimeDistributed(Dense(32,activation='selu'))(output_tensor)
    output_tensor = TimeDistributed(Dense(32,activation='selu'))(output_tensor)
    output_tensor = LSTM(1,activation='selu',return_sequences=False)(input_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='RMSProp',loss='mean_squared_error')
    model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_test,y_test))
    y_pred = model.predict(X_test)
    x = np.arange(y_pred.shape[0])
    plt.plot(x,y_pred)
    plt.plot(x,y_test)
    plt.show()

def zadanie2():
    selected_data = return_selected()
    X,y = make_dataset(selected_data,100,-3)
    train_size = int(X.shape[0]*0.8)
    X_train,y_train = X[:train_size,...],y[:train_size,...]
    X_test,y_test = X[train_size:,...],y[train_size:,...]
    X_train, X_test = normalize_array(X_train,X_test,0)
    y_train, y_test = normalize_array(y_train,y_test,0)
    input_tensor = Input(X_train.shape[1:])
    #Change 2 below to GRU to get GRU NN
    output_tensor = LSTM(1,activation='selu', return_sequences=True)(input_tensor)
    output_tensor = LSTM(1,activation='selu', return_sequences=False)(output_tensor)
    output_tensor = Dense(1,activation='sigmoid')(output_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='RMSProp',loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=32,epochs=1,validation_data=(X_test,y_test))
    y_pred = model.predict(X_test)
    diff_pred = np.diff(y_pred.squeeze())
    diff_true = np.diff(y_test)
    diff_pred = np.sign(diff_pred)
    diff_true = np.sign(diff_true)
    print(confusion_matrix(diff_true,diff_pred))
    print(accuracy_score(diff_true,diff_pred))

if __name__ == '__main__':
    zadanie2()
