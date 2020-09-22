import gc
import random
import nndata
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization
from keras.layers.pooling import AveragePooling2D,MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras import regularizers
from keras.callbacks import ModelCheckpoint

# def create_raw_model(nchan, nclasses, l1=0):
#     """
#     CNN model definition
#     """
#     # input_shape = (trial_length, nchan, 1)
#     input_shape = (960, 64,1)
#     model = Sequential()
#     model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
#     # model.add(AveragePooling2D((4, 4), strides=(1, 1)))
#     model.add(Conv2D(40, (1, 64), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
#     model.add(AveragePooling2D((30, 1), strides=(15, 1)))
#     model.add(Flatten())
#     # model.add(Dense(1024, activation="relu"))
#     model.add(Dense(80, activation="relu"))
#     # model.add(Dense(256, activation="relu"))
#     # model.add(Dense(64, activation="relu"))
#     model.add(Dense(nclasses, activation="softmax"))
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
#     return model
#l2园无
def create_raw_model(nsample, nclasses, l2=0.01):
    """
    CNN model definition
    """
    # input_shape = (trial_length, nchan, 1)
    input_shape = (nsample, 64,1)
    model = Sequential()
    model.add(Conv2D(50, (25, 1), activation="relu", kernel_regularizer=regularizers.l2(l2), padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    # model.add(AveragePooling2D((4, 4), strides=(1, 1)))
    model.add(Conv2D(50, (1, 16), activation="relu", kernel_regularizer=regularizers.l2(l2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Conv2D(25, (30, 1), activation="relu", kernel_regularizer=regularizers.l2(l2), padding="same"))
    model.add(MaxPooling2D((7, 1), strides=(5, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(25, (30, 1), activation="relu", kernel_regularizer=regularizers.l2(l2), padding="same"))
    model.add(MaxPooling2D((3, 1), strides=(2, 1)))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(rate=random.random()/2.0))
    # model.add(Dense(1024, activation="relu", kernel_regularizer=regularizers.l2()))
    # model.add(BatchNormalization())
    # model.add(Dropout(rate=random.random()/2.0))
    model.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=random.random()/2.0))
    model.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=random.random()/2.0))
    model.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=random.random()/2.0))
    model.add(Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=random.random()/2.0))
    model.add(Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.15))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    return model

def create_raw_model2(nchan, nclasses, trial_length=960, l1=0, full_output=False):
    """
    CRNN model definition
    """
    input_shape = (trial_length, nchan, 1)
    model = Sequential()
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))

    model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
    model.add(AveragePooling2D((2, 2), strides=(1, 1)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(40, activation="sigmoid", dropout=0.25, return_sequences=full_output))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
    return model

def fit_model(model, X, y, train_idx, test_idx, input_length=50, batch_size=32, epochs=30, steps_per_epoch=1000, callbacks=None):
    gc.collect()
    return model.fit_generator(
        nndata.crossval_gen(X,y, train_idx, input_length, batch_size),
        validation_data=nndata.crossval_test(X, y, test_idx, input_length),
        steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks
    )
