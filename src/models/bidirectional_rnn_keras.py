import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional


class BidirectionalRNNKeras:
    def __init__(self, num_classes, num_features):
        self.num_features = num_features
        self.num_classes = num_classes

        model = Sequential()

        model.add(Bidirectional(LSTM(3), merge_mode='concat'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def fit(self, X, t, X_val=None, t_val=None, epochs=5, batch_size=20):

        X = X.reshape([X.shape[0], -1, self.num_features])
        t = self._get_one_hot_encoding(t)

        if X_val is not None and t_val is not None:
            X_val = X_val.reshape([X_val.shape[0], -1, self.num_features])
            t_val = self._get_one_hot_encoding(t_val)
            val_data = [X_val, t_val]
        else:
            val_data = None

        return self.model.fit(X, t, epochs=epochs, batch_size=batch_size, validation_data=val_data)

    def score(self, X, t):

        X = X.reshape([X.shape[0], -1, self.num_features])
        t = self._get_one_hot_encoding(t)

        # Evaluate score with the trained model
        score = self.model.evaluate(X, t)

        return score

    def predict(self, X):
        X = X.reshape([X.shape[0], -1, self.num_features])
        return np.argmax(self.model.predict(X), axis=1)

    def _get_one_hot_encoding(self, target):
        onehot = np.zeros([len(target), self.num_classes])
        onehot[np.arange(len(target)), target] = 1
        return onehot


