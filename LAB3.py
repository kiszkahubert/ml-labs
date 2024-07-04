import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def zadanie3_2(data,column,value_to_be_1):
    mask = data[column].values == value_to_be_1
    data.loc[mask,column] = 1
    data.loc[~mask,column] = 0
    return data

def zadanie3_3(data):
    pass

if __name__ == "__main__":
    data = pd.read_excel('files/lab 3/practice_lab_3.xlsx')
    print(data.columns)
    # data_modified = zadanie3_2(data.copy(),"Gender","Female")
    