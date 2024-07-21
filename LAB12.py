import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from tqdm.notebook import tqdm
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10, fashion_mnist, mnist
from keras.models import Model, Sequential
from keras.layers import Lambda, Flatten, Conv2DTranspose, GlobalAveragePooling2D, Dropout, Add, Concatenate, LeakyReLU, Activation, BatchNormalization, GlobalAveragePooling2D, Conv2D, Input, Dense, Reshape, Lambda, Average, UpSampling2D
from keras.optimizers import Adam
from keras.applications import xception, resnet, DenseNet201, VGG16
from keras import backend as K
from matplotlib import pyplot as plt

def zadanie():
    # zadanie 12.1
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1) / 255
    X_test = np.expand_dims(X_test, axis=-1) / 255
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    discriminator = Sequential(
        [
            Lambda(lambda x: K.expand_dims(x, axis=-1)),
            Conv2D(64, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Conv2D(128, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Conv2D(256, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Flatten(),
            Dense(1, activation='sigmoid')
        ],
        name="discriminator"
    )
    discriminator.compile(optimizer=Adam(0.001, beta_1=0.8), loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False

    # zadanie 12.2
    num_new_samples = 10000
    new_data = np.random.randn(num_new_samples, 28, 28)
    new_data = np.expand_dims(new_data, axis=-1) / 255
    new_labels = np.random.randint(0, 10, num_new_samples)
    X_train_extended = np.concatenate((X_train, new_data), axis=0)
    y_train_extended = np.concatenate((y_train, new_labels), axis=0)
    # discriminator.fit(X_train_extended, y_train_extended, epochs=5, batch_size=32)

    # zadanie 12.3
    generator = Sequential(
        [
            Dense(128 * 7 * 7, activation='relu', input_shape=(100,)),
            Reshape((7, 7, 128)),
            BatchNormalization(),
            UpSampling2D(),
            Conv2D(128, kernel_size=(4, 4), padding='same'),
            Activation('relu'),
            BatchNormalization(),
            UpSampling2D(),
            Conv2D(64, kernel_size=(4, 4), padding='same'),
            Activation('relu'),
            BatchNormalization(),
            Conv2D(1, kernel_size=(7, 7), padding='same', activation='sigmoid')
        ],
        name="generator"
    )

    noise_input = Input(shape=(100,))
    generated_image = generator(noise_input)
    discriminator_output = discriminator(generated_image)
    gan = Model(inputs=noise_input, outputs=discriminator_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, beta_1=0.8))

    #to chyba zle jest
    def make_train_batch(X_train, batch_size, hidden_dim, generator):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        noise = np.random.randn(batch_size, hidden_dim)
        gen_imgs = generator.predict(noise)
        X_combined = np.concatenate((real_imgs, gen_imgs))
        y_combined = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))
        return X_combined, y_combined

    def generate_images(img_cnt, hidden_dim, generator):
        noise = np.random.randn(img_cnt, hidden_dim)
        return generator.predict(noise)

    def generate_data(img_cnt, generator, descriptor, classifier, hidden_dim):
        generated_images = generate_images(img_cnt, hidden_dim, generator)
        descriptor_results = descriptor.predict(generated_images)
        classifier_results = classifier.predict(generated_images)
        fig, axes = plt.subplots(1, img_cnt, figsize=(20, 2))
        for i in range(img_cnt):
            axes[i].imshow(generated_images[i, :, :, 0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Desc: {descriptor_results[i][0]:.2f}")
        plt.show()

        return generated_images, descriptor_results, classifier_results

    img_cnt = 10  
    hidden_dim = 100
    epoch_cnt = 10
    batch_cnt = 256
    batch_size = 64

    for i in range(epoch_cnt):
        print(f'epoch {i+1} of {epoch_cnt}')
        for batch in tqdm(range(batch_cnt)):
            X_all, y_all = make_train_batch(X_train, batch_size, hidden_dim, generator)
            discriminator.train_on_batch(X_all, y_all)
            X_gan = np.random.randn(batch_size * 2, hidden_dim)
            y_gan = np.ones((batch_size * 2,))
            gan.train_on_batch(X_gan, y_gan)
        
        generate_data(10, generator, discriminator, discriminator, hidden_dim)

if __name__ == '__main__':
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    zadanie()
