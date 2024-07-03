import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

bh_data = pd.read_excel('files/lab 2/practice_lab_2.xlsx')
bh_arr = bh_data.values
x,y = bh_arr[:,:-1], bh_arr[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=221,shuffle=False)

linReg = LinearRegression()
linReg.fit(x_train,y_train)
y_pred = linReg.predict(x_test)
minval = min(y_test.min(),y_pred.max())
maxval = max(y_test.max(),y_pred.min())
plt.scatter(y_test,y_pred)
plt.plot([minval,maxval],[minval,maxval])
plt.xlabel('y_test')
plt.xlabel('y_pred')
plt.show()