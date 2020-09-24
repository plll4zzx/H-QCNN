#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = {
    'name': 'Zhaoxi Zhang',
    'Email': 'zhaoxi_zhang@163.com',
    'QQ': '809536596',
    'Created': ''
}

import numpy as np
from tensorflow import keras
import os

def select_data(
        data_x,
        data_y,
        num_class,
        num_data
):
    x=[]
    y=[]
    for idx in range(data_x.shape[0]):
        if data_y[idx]==num_class:
            x.append(data_x[idx])
            y.append(data_y[idx])
        if len(x)==num_data:
            break
    return x,y


def get_data(num_class=[0,1,2,3]):
    (x_train,y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    channels, img_rows, img_cols=1,28,28
    num_classes=10
    # if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
    # else:
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    #     input_shape = (img_rows, img_cols, channels)

    x_train = x_train.astype('float32') / 127.5 - 1.  # /128-1
    x_test = x_test.astype('float32') / 127.5 - 1.

    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    # y_train=y_train.astype('long')
    # y_test=y_test.astype('long')
    # num_class=num_class=[0,1,2,3]
    num_train=100
    num_test=100
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    state=0
    for nc in num_class:
        temp_x, temp_y=select_data(x_train, y_train, nc, num_train)
        train_x=train_x+temp_x
        train_y=train_y+temp_y
    for nc in num_class:
        temp_x, temp_y=select_data(x_test, y_test, nc, num_test)
        test_x=test_x+temp_x
        test_y=test_y+temp_y

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return train_x, train_y, test_x, test_y
if __name__=='__main__':
    # cg=CIFAR10_Gray(standardization=True)
    # plt.imshow(cg.x_train[1000,:,:,0], cmap='gray')
    # plt.show()
    # print()

    c=CIFAR10(standardization=False)
    a=2200
    b=[]
    for a in range(100,200,1):
        if c.y_train[a][2]==1:
            b.append(c.x_train[a,:,:,:])
            # mean=c.x_train[a,:,:,:]*0.5+0.5
            # plt.imshow(mean)
            # plt.show()
    mean=np.mean(b, axis=0)*0.5+0.5
    plt.imshow(mean)
    plt.show()
    print()
