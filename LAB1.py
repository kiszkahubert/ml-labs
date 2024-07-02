from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel("files/lab 1/practice_lab_1.xlsx")
nazwy_kolumn = list(data.columns)
wartosci_kolumn = np.array(data.values)
np.set_printoptions(suppress=True,precision=2)

def zadanie1_2():
    #1.2.1
    res = wartosci_kolumn[::2,:] - wartosci_kolumn[1::2,:]
    #1.2.2


zadanie1_2()