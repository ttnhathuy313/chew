import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from loguru import logger
import sys
import tensorflow as tf

# TODO: whats wrong with verbose here?
def create_model(dataset_shape=(None, 80, 17), verbose=20):
    logger.add(sys.stdout, level=verbose)
    model = Sequential()
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.build(input_shape=dataset_shape)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    logger.info(f"Model: {model.summary()}")
    model.save("./conv1d-model.keras")
    logger.info(f"Saved model to ./conv1d-model.keras")
    return model

def fit(model, dataset, epochs=10, verbose=20):
    logger.add(sys.stdout, level=verbose)
    logger.info(f"Fitting model for {epochs} epochs")
    model.fit(dataset, epochs=epochs)
    logger.info(f"Finished fitting model for {epochs} epochs")
    model.save("./conv1d-model.keras")
    logger.info(f"Saved model to ./conv1d-model.keras")