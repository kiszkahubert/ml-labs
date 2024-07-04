import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree

def zadanie3_2(data,column,value_to_be_1):
    mask = data[column].values == value_to_be_1
    data.loc[mask,column] = 1
    data.loc[~mask,column] = 0
    return data

def zadanie3_3(data,column):
    one_hot = pd.get_dummies(data[column])
    data = pd.concat([data,one_hot],axis=1)
    data = data.drop(columns=[column])
    return data

def zadanie3_4(TP,FN,FP,TN):
    sensitivity = TP/(TP+FN)
    precision = TP/(TP+FP)
    specificity = TN/(FP+TN)
    accuracy = (TP+TN)/(TP+FN+FP+TN)
    return sensitivity, precision, specificity, accuracy

def zadanie3_5(data):
    bh_arr = data.values
    x,y = bh_arr[:,:-1], bh_arr[:,-1]
    y = y.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=41)
    scaler = RobustScaler() #tu jest zadanie 3.6 MinMaxScaler najlepszy
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    models = [kNN(), SVC()] 
    for model in models:
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test,y_pred)
        print(f"Confusion Matrix for {model.__class__.__name__}:\n{cm}\n")

def zadanie3_7():
    data = load_breast_cancer()
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    dtree = DT(max_depth=5,random_state=42)
    dtree.fit(x_train,y_train)
    plt.figure(figsize=(20,10))
    plot_tree(dtree,feature_names=data.feature_names, class_names=data.target_names, filled=True)
    plt.show()


if __name__ == "__main__":
    pd.set_option('display.max_columns',None)
    data = pd.read_excel('files/lab 3/practice_lab_3.xlsx')
    data = zadanie3_2(data,'Gender','Female')
    data = zadanie3_2(data,'Married','Yes')
    data = zadanie3_2(data,'Education','Graduate')
    data = zadanie3_2(data,'Self_Employed','Yes')
    data = zadanie3_2(data,'Loan_Status','Y')
    data = zadanie3_3(data,'Dependents')
    data = zadanie3_3(data,'Property_Area')
    zadanie3_5(data)