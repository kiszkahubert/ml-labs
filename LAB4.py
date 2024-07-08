import pandas as pd
import os
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler



def zadanie4_1(path):
    fs, audio_data = wavfile.read(path,mmap=True)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    x_raw = np.zeros((1, len(audio_data)))
    x_raw[0, :len(audio_data)] = audio_data
    x_fft = np.abs(fft(x_raw, axis=1)) / x_raw.shape[1]
    resolution = fs / len(audio_data)
    mean_num = max(1,int(round(resolution)))
    x_fft = np.reshape(x_fft, (x_fft.shape[0], x_fft.shape[1] // mean_num, mean_num))
    x_fft = x_fft.mean(axis=-1)
    low_cut = 50
    high_cut = 280
    x_fft_cut = x_fft[:, low_cut:high_cut]
    x_fft_cut /= np.expand_dims(x_fft_cut.max(axis=1), axis=-1)
    # freq_axis = np.linspace(low_cut, high_cut, x_fft_cut.shape[1])
    # plt.figure(figsize=(8, 6))
    # plt.plot(freq_axis, x_fft_cut[0, :])
    # plt.xlim(low_cut, high_cut)
    # plt.grid(True)
    # plt.show()
    return x_fft_cut[0,:]

def zadanie4_2():
    paths = ['files/voices/man_sample_1.wav','files/voices/man_sample_2.wav','files/voices/man_sample_3.wav','files/voices/man_sample_4.wav',\
             'files/voices/woman_sample_1.wav','files/voices/woman_sample_2.wav','files/voices/woman_sample_3.wav','files/voices/woman_sample_4.wav',]
    labels = [0,0,0,0,1,1,1,1]
    features = []
    for path in paths:
        features.append(zadanie4_1(path))

    x, y = np.array(features), np.array(labels)
    conf_matrix_knn = np.zeros((2,2))
    conf_matrix_svm = np.zeros((2,2))
    conf_matrix_dt = np.zeros((2,2))
    for _ in range(30):
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
        scaler = StandardScaler() #sluzy do przeksztalcania danych tak aby mialy srednia 0 i odchylenie standardowe 1
        x_train = scaler.fit_transform(x_train) 
        x_test = scaler.transform(x_test)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(x_train,y_train)
        y_pred_knn = knn.predict(x_test)
        conf_matrix_knn += confusion_matrix(y_test,y_pred_knn, labels=[0,1])

        svm = SVC()
        svm.fit(x_train,y_train)
        y_pred_svm = svm.predict(x_test)
        conf_matrix_svm += confusion_matrix(y_test,y_pred_svm, labels=[0,1])

        dt = DecisionTreeClassifier()
        dt.fit(x_train,y_train)
        y_pred_dt = dt.predict(x_test)
        conf_matrix_dt += confusion_matrix(y_test,y_pred_dt, labels=[0,1])

    conf_matrix_knn /= 30
    conf_matrix_svm /= 30
    conf_matrix_dt /= 30
    print("Wyniki kNN:")
    print(conf_matrix_knn)
    print("Wyniki SVM:")
    print(conf_matrix_svm)
    print("Wyniki Decision Tree:")
    print(conf_matrix_dt)

def zadanie4_3():
    #TODO nie ma tego pliku nigdzie takze nie zrobie tego
    pass

#zadanie 4.4
class featuresCountBasedOnVariation(BaseEstimator, TransformerMixin):
    def __init__ (self,explained_variance_ratio=0.95):
        self.explained_variance_ratio = explained_variance_ratio
        self.pca = None
        self.n_components = None
    
    def fit(self, x, y=None):
        self.pca = PCA()
        self.pca.fit(x)
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components = np.argmax(cumulative_variance >= self.explained_variance_ratio) + 1
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(x)
        return self
    
    def transform(self,x):
        return self.pca.transform(x)
    
    def fit_transform(self,x,y=None):
        self.fit(x,y)
        return self.transform(x)




if __name__ == "__main__":
    zadanie4_2()