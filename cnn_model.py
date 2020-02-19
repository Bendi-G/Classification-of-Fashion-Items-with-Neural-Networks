from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # pylint: disable=unused-argument
        # pylint: disable=unused-variable
        return self

    def transform(self, X, y=None):
        # pylint: disable=unused-argument
        # pylint: disable=unused-variable
        # reshape data to have a single channel
        X = X.reshape((X.shape[0], 28, 28, 1))

        # Turn our images into floating point numbers
        X = X.astype('float32')

        # Put our input data in the range 0-1
        X /= 255

        if y is None:
            return X
        # Convert class vectors to binary class matrices
        y = np_utils.to_categorical(y, 4)
        return X, y


def keras_builder():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='Same', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.35))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=4, activation='softmax'))

    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_model():
    preprocessor = Preprocessor()

    model = KerasClassifier(build_fn=keras_builder, batch_size=100, epochs=50)

    return Pipeline([("preprocessor", preprocessor), ("model", model)])
