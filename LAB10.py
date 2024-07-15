import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.datasets import fashion_mnist, mnist
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D,GlobalAveragePooling2D,Dense,Input,Reshape,UpSampling2D,BatchNormalization,GaussianNoise
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot as plt


def test():