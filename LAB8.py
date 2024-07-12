import pandas as pd
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, BatchNormalization, Lambda
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from matplotlib import rcParams
from PIL import Image

class tests:

    def show_pictures(arrs):
        arr_cnt = arrs.shape[0]
        fig, axes = plt.subplots(1,arr_cnt,figsize=(5*arr_cnt,arr_cnt))
        for axis, pic in zip (axes,arrs):
            axis.imshow(pic.squeeze(),cmap='gray')
            axis.axis('off')
        fig.tight_layout()
        return fig

    def main():
        (X_train,y_train), (X_test,y_test) = fashion_mnist.load_data()
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        y_train = pd.get_dummies(pd.Categorical(y_train)).values
        y_test = pd.get_dummies(pd.Categorical(y_test)).values
        rcParams['font.size'] = 48
        #rotated
        rotated_images = X_train[:10].copy()
        img_size = X_train.shape[1:3]
        # angles = np.random.randint(-30,30,len(rotated_images))
        # for i, img in enumerate(rotated_images):
        #     angle = np.random.randint(-30,30)
        #     img = img.squeeze()
        #     img = Image.fromarray(img).rotate(angle,expand=1).resize(img_size)
        #     rotated_images[i] = np.expand_dims(np.array(img),axis=-1)
        
        # fig = show_pictures(rotated_images)
        # plt.show()

        #croped
        for i, img in enumerate(rotated_images):
            angle = np.random.randint(-30,30)
            left, upper = np.random.randint(0,5,2)
            right, lower = np.random.randint(23,28,2)
            img = img.squeeze()
            img = Image.fromarray(img).crop((left,upper,right,lower)).resize(img_size)
            rotated_images[i] = np.expand_dims(np.array(img),axis=-1)
        
        fig = show_pictures(rotated_images)
        plt.show()

    def encoder_decoder():
        

if __name__ == '__main__':
    test = tests()
