import keras.layers
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10, fashion_mnist, mnist
from keras.models import Model, Sequential
from keras.layers import Flatten,Conv2DTranspose,GlobalAveragePooling2D,Dropout,Add,Concatenate,LeakyReLU,Activation,BatchNormalization,GlobalAveragePooling2D,Conv2D, Input, Dense, Reshape, Lambda, Average
from keras.optimizers import Adam
from keras.applications import xception, resnet, DenseNet201, VGG16
from matplotlib import pyplot as plt

def zadanie11_1(X_train,X_test,y_train,y_test):
    cls_cnt = 10
    input_tensor = Input(X_train.shape[1:])
    base_model = xception.Xception(include_top=False)
    output_tensor = Lambda(lambda x: xception.preprocess_input(x))(input_tensor)
    output_tensor = base_model(output_tensor)
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(cls_cnt,activation='softmax')(output_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    base_model.trainable = True
    model.fit(X_train,y_train,epochs=10,batch_size=32)
    base_model.trainable = False
    #nie wiem czy przed nie trzeba jeszcze raz skompilowac
    model.fit(X_train,y_train,epochs=10,batch_size=32)
    _, acc = model.evaluate(X_test,y_test)
    print(f"Accuracy: {acc:.4f}")

def zadanie11_2(X_train,X_test,y_train,y_test):
    cls_cnt = 10
    input_tensor = Input(shape=X_train.shape[1:])
    xception_base = xception.Xception(include_top=False,input_tensor=input_tensor)
    xception_output = Flatten()(xception_base.output)
    resnet_base = resnet.ResNet50(include_top=False,input_tensor=input_tensor)
    resnet_output = Flatten()(resnet_base.output)
    densenet_base = DenseNet201(include_top=False,input_tensor=input_tensor)
    densenet_output = Flatten()(densenet_base.output)
    combined = Concatenate()([xception_output,resnet_output,densenet_output])
    output_tensor = Dense(cls_cnt,activation='softmax')(combined)
    model = Model(inputs=input_tensor,outputs=output_tensor)
    model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    xception_base.trainable = True
    resnet_base.trainable = True
    densenet_base.trainable = True
    model.fit(X_train,y_train,epochs=10,batch_size=32)

#-------------------------------- it aint working -----------------------------------------
# def zadanie11_3():
#     (X_train,y_train),(_,_) = cifar10.load_data()
#     X_train = X_train[y_train.flatten() == 3]
#     X_train = X_train.astype('float32') / 127.5 - 1.0
#     input_shape = (32,32,3)
#     discriminator = Sequential(
#         [
#             Conv2D(64,kernel_size=4,strides=2,padding='same',input_shape=input_shape),
#             LeakyReLU(alpha=0.2),
#             Dropout(0.25),
#             Conv2D(128,kernel_size=4,strides=2,padding='same'),
#             LeakyReLU(alpha=0.2),
#             Dropout(0.25),
#             Conv2D(256, kernel_size=4, strides=2, padding='same'),
#             LeakyReLU(alpha=0.2),
#             Dropout(0.25),
#             Flatten(),
#             Dense(1,activation='sigmoid')
#         ],
#         name="discriminator"
#     )
#     generator = Sequential(
#         [
#             Dense(256*4*4,activation='relu'),
#             Reshape((4,4,256)),
#             BatchNormalization(),
#             Conv2DTranspose(256, kernel_size=4,strides=2,padding='same'),
#             Activation('relu'),
#             BatchNormalization(),
#             Conv2DTranspose(128,kernel_size=4,strides=2,padding='same'),
#             Activation('relu'),
#             BatchNormalization(),
#             Conv2DTranspose(64,kernel_size=4,strides=2,padding='same'),
#             Activation('relu'),
#             BatchNormalization(),
#             Conv2DTranspose(3,kernel_size=4,strides=2,padding='same'),
#             Activation('tanh')
#         ],
#         name="generator"
#     )
#     discriminator.compile(optimizer=Adam(0.0001),loss='binary_crossentropy',metrics=['accuracy'])
#     discriminator.trainable=False
#     z = Input(shape=(100,))
#     img = generator(z)
#     validity = discriminator(img)
#     gan = Model(z,validity)
#     gan.compile(optimizer=Adam(0.0001),loss='binary_crossentropy')
#     def train(epochs, batch_size=128, save_interval=50):
#         real = np.ones((batch_size, 1))
#         fake = np.zeros((batch_size, 1))
#         for epoch in range(epochs):
#             idx = np.random.randint(0, X_train.shape[0], batch_size)
#             imgs = X_train[idx]
#             noise = np.random.normal(0, 1, (batch_size, 100))
#             gen_imgs = generator.predict(noise)
#             d_loss_real = discriminator.train_on_batch(imgs, real)
#             d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
#             d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#             noise = np.random.normal(0, 1, (batch_size, 100))
#             g_loss = gan.train_on_batch(noise, real)
#             print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss}]")
#             if epoch % save_interval == 0:
#                 save_imgs(epoch)

#     def save_imgs(epoch):
#         r, c = 5, 5
#         noise = np.random.normal(0, 1, (r * c, 100))
#         gen_imgs = generator.predict(noise)
#         gen_imgs = 0.5 * gen_imgs + 0.5
#         fig, axs = plt.subplots(r, c)
#         cnt = 0
#         for i in range(r):
#             for j in range(c):
#                 axs[i, j].imshow(gen_imgs[cnt, :, :, :])
#                 axs[i, j].axis('off')
#                 cnt += 1
#         fig.savefig("cat_%d.png" % epoch)
#         plt.close()

#     train(epochs=10000, batch_size=64, save_interval=1000)

if __name__ == '__main__':
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    (X_train,y_train),(X_test,y_test) = cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)
    # zadanie11_1(X_train,X_test,y_train,y_test)
    zadanie11_3()