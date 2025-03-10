import tensorflow as tf
import pandas as pd
import numpy as np
from keras.datasets import fashion_mnist, mnist
from keras.models import Model, Sequential
from keras.layers import Flatten,MaxPooling2D,Conv2D, Input, Dense, Reshape, BatchNormalization, Lambda
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot as plt
from matplotlib import rcParams
from PIL import Image
from tensorflow.image import resize  # type: ignore

class tests:

    @staticmethod
    def show_pictures(arrs):
        arr_cnt = arrs.shape[0]
        fig, axes = plt.subplots(1,arr_cnt,figsize=(5*arr_cnt,arr_cnt))
        for axis, pic in zip (axes,arrs):
            axis.imshow(pic.squeeze(),cmap='gray')
            axis.axis('off')
        fig.tight_layout()
        plt.show()
        return fig

    def main(self):
        (X_train,y_train), (X_test,y_test) = fashion_mnist.load_data()
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        y_train = pd.get_dummies(pd.Categorical(y_train)).values
        y_test = pd.get_dummies(pd.Categorical(y_test)).values
        rcParams['font.size'] = 48
        #rotated
        rotated_images = X_train[:10].copy()
        img_size = X_train.shape[1:3]
        angles = np.random.randint(-30,30,len(rotated_images))
        for i, img in enumerate(rotated_images):
            angle = np.random.randint(-30,30)
            img = img.squeeze()
            img = Image.fromarray(img).rotate(angle,expand=1).resize(img_size)
            rotated_images[i] = np.expand_dims(np.array(img),axis=-1)
        
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
        
        fig = self.show_pictures(rotated_images)
        plt.show()
    
    @staticmethod
    def load_data():
        (X_train,y_train), (X_test,y_test) = mnist.load_data()
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        y_train = pd.get_dummies(pd.Categorical(y_train)).values
        y_test = pd.get_dummies(pd.Categorical(y_test)).values
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def adding_noise(tensor):
        noise = K.random_normal(shape=(K.shape(tensor)),mean=0, stddev=1.5)
        return tensor + noise
    
    @staticmethod
    def filter_data(data,iteration_num,autoencoder):
        augmented_data = data.copy()
        for i in range(iteration_num):
            augmented_data = autoencoder.predict(augmented_data)
        return augmented_data
        
    def encoder_decoder(self):
        X_train, X_test, y_train, y_test = self.load_data()
        #encoder
        act_func = 'selu'
        hidden_dims = 64
        encoder_layers = [
            Reshape((28*28,)),
            BatchNormalization(),
            Dense(512,activation=act_func),
            Dense(128,activation=act_func),
            Dense(64,activation=act_func),
            Dense(hidden_dims,activation=act_func)]
        tensor = encoder_input = Input(shape=(28,28,1))
        for layer in encoder_layers:
            tensor = layer(tensor)

        encoder_output = tensor
        encoder = Model(inputs=encoder_input, outputs=encoder_output)
        #decoder
        decoder_layers = [
            Dense(128,activation=act_func),
            Dense(512,activation=act_func),
            Dense(784,activation='sigmoid'),
            Reshape((28,28,1)),
            Lambda(lambda x: x*255)]
        decoder_input = tensor = Input(encoder_output.shape)
        for layer in decoder_layers:
            tensor = layer(tensor)
        
        decoder_output = tensor
        decoder = Model(inputs=decoder_input, outputs=decoder_output)
        #autoencoder
        aec_output = decoder(encoder(encoder_input))
        gen_autoencoder = Model(inputs = encoder_input, outputs = aec_output)
        gen_autoencoder.compile(optimizer=Adam(0.0001),loss='MeanSquaredError')
        gen_autoencoder.fit(x=X_train,y=X_train, validation_data=(X_test,X_test),batch_size=32,epochs=10)
        #noise
        noised_encoder_output = Lambda(self.adding_noise)(encoder_output)
        augmenter_output = decoder(noised_encoder_output)
        augmenter = Model(inputs=encoder_input, outputs=augmenter_output)
        #generation
        start = 50
        end = start + 10
        for i in range(10):
            test_for_augm = X_train[i*10:i*10+10,...]
            augmented_data = test_for_augm.copy()
            self.show_pictures(test_for_augm)
            augmented_data = augmenter.predict(augmented_data)
            self.show_pictures(augmented_data)
            augmented_data = self.filter_data(augmented_data,5,gen_autoencoder)
            self.show_pictures(augmented_data)

class task:
    @staticmethod
    def build_model(X_train,num_layers,class_cnt):
        filter_cnt = 32
        learning_rate = 0.0001
        act_func = 'relu'
        conv_rule = 'same'
        kernel_size = (3,3)
        model = Sequential()
        model.add(Conv2D(input_shape=X_train.shape[1:],
                         filters=filter_cnt,
                         kernel_size=kernel_size,
                         padding=conv_rule,
                         activation=act_func))
        for _ in range(num_layers):
            model.add(Conv2D(filters=filter_cnt,
                             kernel_size=kernel_size,
                             padding=conv_rule,
                             activation=act_func))
        model.add(Flatten())
        model.add(Dense(class_cnt, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate),loss='SparseCategoricalCrossentropy',metrics=['accuracy'])
        return model
    
    @staticmethod
    def rotate_image(img):
        angle = np.random.randint(-30,30)
        img = Image.fromarray(img.squeeze()).rotate(angle,expand=1)
        img = np.expand_dims(np.array(img), axis=-1)
        return img
    
    @staticmethod
    def flip_image(img):
        return img[...,::-1,:]

    @staticmethod
    def scale_image(img):
        return resize(img,(28,28))

    @staticmethod
    def add_noise(img):
        noise = K.random_normal(shape=(K.shape(img)),mean=0,stddev=1.5)
        return img + noise
    
    @staticmethod
    def create_autoencoder(X_train,X_test):
        act_func = 'selu'
        hidden_dims = 64
        encoder_layers = [
            Reshape((28*28,)),
            BatchNormalization(),
            Dense(512,activation=act_func),
            Dense(128,activation=act_func),
            Dense(64,activation=act_func),
            Dense(hidden_dims,activation=act_func)]
        tensor = encoder_input = Input(shape=(28,28,1))
        for layer in encoder_layers:
            tensor = layer(tensor)

        encoder_output = tensor
        encoder = Model(inputs=encoder_input, outputs=encoder_output)
        #decoder
        decoder_layers = [
            Dense(128,activation=act_func),
            Dense(512,activation=act_func),
            Dense(784,activation='sigmoid'),
            Reshape((28,28,1)),
            Lambda(lambda x: x*255)]
        decoder_input = tensor = Input(encoder_output.shape)
        for layer in decoder_layers:
            tensor = layer(tensor)
        
        decoder_output = tensor
        decoder = Model(inputs=decoder_input, outputs=decoder_output)
        #autoencoder
        aec_output = decoder(encoder(encoder_input))
        gen_autoencoder = Model(inputs = encoder_input, outputs = aec_output)
        gen_autoencoder.compile(optimizer=Adam(0.0001),loss='MeanSquaredError')
        gen_autoencoder.fit(x=X_train,y=X_train, validation_data=(X_test,X_test),batch_size=32,epochs=500)
        return gen_autoencoder
    
    def execute_model(self):
        (X_test, y_test), (X_train, y_train) = fashion_mnist.load_data()
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        X_train_augmented = []
        y_train_augmented = []
        autoencoder = self.create_autoencoder(X_train,X_test)
        genererated_data = autoencoder.predict(X_train)
        X_train_augmented = np.concatenate([X_train,genererated_data],axis=0)
        y_train_augmented = np.concatenate([y_train,y_train],axis=0)
        #mozliwe ze tu brakuje fragmentu kodu ale nw
        model = self.build_model(X_train_augmented, 4, np.max(y_train) + 1)
        model.fit(x=X_train_augmented, y=y_train_augmented, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=2)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        #task 1
        # for img, label in zip(X_train, y_train):
        #     X_train_augmented.append((img, label))
        #     X_train_augmented.append((self.rotate_image(img), label))
        #     X_train_augmented.append((self.flip_image(img), label))
        #     X_train_augmented.append((self.add_noise(img), label))

        # X_train_augmented = [(self.scale_image(img), label) for img, label in X_train_augmented]
        # X_train_augmented, y_train_augmented = zip(*X_train_augmented)
        # X_train_augmented = np.array(X_train_augmented)
        # y_train_augmented = np.array(y_train_augmented)
        # model = self.build_model(X_train_augmented, 4, np.max(y_train) + 1)
        # model.fit(x=X_train_augmented, y=y_train_augmented, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=2)
        # test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        # print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        # print(f"Test Loss: {test_loss:.4f}")
        #task 2


if __name__ == '__main__':
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    task = task()
    task.execute_model()