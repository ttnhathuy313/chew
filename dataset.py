import tensorflow as tf
import numpy as np
import pandas as pd
from loguru import logger
import sys


class Dataset(object):
    def __init__(self, paths, window_size, batch_size, shuffle=True, buffer_size=1000, verbose=20):
        logger.remove(0)
        logger.add(sys.stdout, level=verbose)
        logger.info(f"Building dataset with window_size={window_size}, batch_size={batch_size}, shuffle={shuffle}, buffer_size={buffer_size}")
        self.paths = paths
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self._load_data()
        self._build_dataset()
        logger.info(f"Built dataset: {self.dataset}")

    def _load_data(self):
        assert isinstance(self.paths, dict)
        positive_paths = self.paths['positive']
        negative_paths = self.paths['negative']

        from numpy.lib.stride_tricks import as_strided
        def rolling_window(a, window):
            shape = (a.shape[0] - window + 1, window) + a.shape[1:]
            strides = (a.strides[0],) + a.strides
            return as_strided(a, shape=shape, strides=strides)

        positive_data = np.array([])
        for path in positive_paths:
            if path in negative_paths:
                logger.warning(f"Path {path} is in both positive and negative paths")
                continue
            data = pd.read_csv(path).values
            # add label
            data = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
            # get tensors of 3-second window
            data = rolling_window(data, self.window_size)
            # append to positive_data
            if positive_data.size == 0:
                positive_data = data
            else:
                positive_data = np.concatenate((positive_data, data), axis=0)
        logger.trace(f"Positive data shape: {positive_data.shape}")

        negative_data = np.array([])
        for path in negative_paths:
            if path in positive_paths:
                logger.warning(f"Path {path} is in both positive and negative paths")
                continue
            data = pd.read_csv(path).values
            data = np.concatenate((data, np.zeros((data.shape[0], 1))), axis=1)
            # get tensors of 3-second window
            data = rolling_window(data, self.window_size)
            # append to positive_data
            if negative_data.size == 0:
                negative_data = data
            else:
                negative_data = np.concatenate((negative_data, data), axis=0)
        logger.trace(f"Negative data shape: {negative_data.shape}")

        # randomly choose positive_data so that positive_data and negative_data have the same size
        # use seed to make sure that the same data is chosen
        np.random.seed(0)
        if (positive_data.shape[0] > negative_data.shape[0]):
            positive_data = positive_data[np.random.choice(positive_data.shape[0], negative_data.shape[0], replace=False)]
        if (negative_data.shape[0] > positive_data.shape[0]):
            negative_data = negative_data[np.random.choice(negative_data.shape[0], positive_data.shape[0], replace=False)]

        # merge into one
        data = np.concatenate((positive_data, negative_data), axis=0)
        # extract labels
        self.labels = data[:, :, -1][:,0]
        logger.trace(f"Labels shape: {self.labels.shape}")
        data = data[:, :, :-1]
        self.data = data
        logger.trace(f"Data shape: {self.data.shape}")

    def _normalize(self):
        # normalize data
        # TODO: use better normalization
        self.data = self.data / np.linalg.norm(self.data, axis=1, keepdims=True)

    def _build_dataset(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.data, self.labels))
        logger.trace(f"Dataset: {self.dataset}")
        if self.shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=self.buffer_size)
        self.dataset = self.dataset.batch(self.batch_size)
        logger.trace(f"Batch Dataset: {self.dataset}")
        self.dataset = self.dataset.prefetch(1)

    

if __name__ == '__main__':

    paths = {
        "positive": [
            "./data/Test_1BEatHeadStill.csv",
            "./data/Test_1CChewGuava.csv",
            "./data/Test_2AEatMoving.csv",
            "./data/Test_2CEatNhan.csv",
        ],
        "negative": [
            "./data/Test_1ANoeatHeadStill.csv",
            "./data/Test_2ANoeatMoving.csv",
        ]
    }
    dataset = Dataset(paths, window_size=80, batch_size=32, verbose=5).dataset
    for data, label in dataset.take(2):
        logger.trace(f"Sanity check: {data.shape}, {label.shape}")