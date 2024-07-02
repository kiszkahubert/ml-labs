from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel("files/lab 1/practice_lab_1.xlsx")
nazwy_kolumn = list(data.columns)
wartosci_kolumn = np.array(data.values)
np.set_printoptions(suppress=True,precision=2)
print(nazwy_kolumn)
print(wartosci_kolumn)