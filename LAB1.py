from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel("files/lab 1/practice_lab_1.xlsx")
nazwy_kolumn = list(data.columns)
wartosci = np.array(data.values)
np.set_printoptions(suppress=True,precision=2)

def zadanie1_2():
    #1.2.1
    res1 = wartosci[::2, :] - wartosci[1::2, :]
    #1.2.2
    res2 = (wartosci - wartosci.mean())/wartosci.std()
    #1.2.3
    res3 = (wartosci - wartosci.mean(axis=0))/(wartosci.std(axis=0)+np.spacing(wartosci.std(axis=0)))
    #1.2.4
    res4 = (wartosci.mean(axis=0)/(wartosci.std(axis=0)+np.spacing(wartosci.std(axis=0))))
    #1.2.5
    max_index = np.argmax(res4)
    max_value = res4[max_index]
    #1.2.6
    mean_cols = wartosci.mean(axis=0)
    gt_mean = wartosci > mean_cols
    count_gt = gt_mean.sum(axis=0)
    #1.2.7
    max_val = wartosci.max()
    cols_with_max = wartosci == max_val
    cols_with_true = np.any(cols_with_max,axis=0)
    cols_index = np.where(cols_with_true)[0]
    #print([nazwy_kolumn[i] for i in cols_index])
    #1.2.8
    cols_masked = wartosci == 0
    count_of_zero_in_cols = np.sum(cols_masked, axis=0)
    max_index = np.argmax(count_of_zero_in_cols)
    # print(nazwy_kolumn[max_index])
    #1.2.9
    cols = wartosci[::2,].sum(axis=0) > wartosci[1::2,].sum(axis=0)
    idx = np.where(cols)[0]
    print([nazwy_kolumn[i] for i in idx])


zadanie1_2()