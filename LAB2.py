import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def zadanie2_1(data):
    corr_of_data = data.corr()
    #print(corr_of_data)
    dependent_var = 'MedianowaCena'
    independent_var = [col for col in data.columns if col != dependent_var]
    num_plots = len(independent_var)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, 5*num_rows),squeeze=False,sharex='col',sharey="row",gridspec_kw={'hspace':0.5,'wspace':0.3})
    for i, var in enumerate(independent_var):
        row = i // num_cols
        col = i % num_cols
        ax[row, col].scatter(data[var], data[dependent_var], alpha=0.5)
        ax[row, col].set_xlabel(var)
        ax[row, col].set_ylabel(dependent_var)
        ax[row, col].grid(True)
    
    if num_plots % num_cols != 0:
        for j in range(num_cols - num_plots % num_cols):
            fig.delaxes(ax[num_rows - 1, num_plots % num_cols + j])
    
    plt.autoscale()
    plt.show()

def zadanie2_2(data, repeat):
    bh_arr = data.values
    x,y = bh_arr[:,:-1], bh_arr[:,-1]
    avg_error = 0
    for i in range(repeat):
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=False)
        linReg = LinearRegression()  
        linReg.fit(x_train,y_train)
        y_pred = linReg.predict(x_test)
        mse = mean_squared_error(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mape = mean_absolute_percentage_error(y_test,y_pred)
        avg_error += mape
    
    return avg_error/repeat

def zadanie2_3():




if __name__ == "__main__":
    bh_data = pd.read_excel('files/lab 2/practice_lab_2.xlsx')
    print(zadanie2_2(bh_data,3))