import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

bh_data = pd.read_excel('/files/lab2/practice_lab_2.xlsx')
bh_arr = bh_data.values
x,y = bh_arr[:,:-1], bh_arr[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=221,shuffle=False)
