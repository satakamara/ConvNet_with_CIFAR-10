import numpy as np
import os
from keras.datasets import cifar10
from keras import backend as K

def load_data(training_samples):
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(X_train, y_train), (X_test, y_test)`.
    """



    print('cifar10')
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    X_train = X_train[:training_samples]
    y_train = y_train[:training_samples]

    return (X_train, y_train), (X_test, y_test)
