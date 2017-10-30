import pickle

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.utils.generic_utils import get_custom_objects

from common.utils import haversine
from models import Model
from config import DATA_DIR
from keras.layers import Activation, Dense, LSTM


class RNNModel(Model):
    def __init__(self):
        self.metadata = None
        self.clusters = None
        self.lr = 0.01
        self.momentum = 0.9
        self.clip_value = 1.
        self.model = None
        self.embedding_dim = 10
        self.batch_size = 200
        self.n_epochs = 1
        self.X_train, self.Y_train = None, None
        self.X_valid, self.Y_valid = None, None
        self.X_test, self.Y_test = None, None

    def set_data(self, data):
        self.X_train, self.Y_train = data.X_train, data.Y_train
        self.X_valid, self.Y_valid = data.X_valid, data.Y_valid
        self.X_test, self.Y_test = data.X_test, data.Y_test

    @staticmethod
    def start_new_session():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.Session(config=config, graph=tf.get_default_graph())
        K.tensorflow_backend.set_session(session)

    def predict(self, features):
        self.model.predict(features)

    def save(self, model_prefix='model'):
        model_json = self.model.to_json()
        with open(DATA_DIR + "cache/model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(DATA_DIR + "cache/model.h5")
        print("Saved model to disk")

    def load(self, model_prefix='latest'):
        json_file = open(DATA_DIR + 'cache/{}-model.json'.format(model_prefix), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(DATA_DIR + "cache/{}-model.h5".format(model_prefix))
        print("Loaded model from disk")

        return loaded_model

    def create_model(self):
        model = Sequential()
        model.add(LSTM(500, input_shape=(10, 2)))

        model.add(Dense(len(self.clusters)))
        model.add(Activation('softmax'))

        cast_clusters = K.cast_to_floatx(self.clusters)

        def distance(probabilities):
            return tf.matmul(probabilities, cast_clusters)

        get_custom_objects().update({'distance': Activation(distance)})
        model.add(Activation(distance))
        optimizer = SGD(lr=0.01, momentum=0.9, clipvalue=1.)
        model.compile(loss=haversine(tensor=True), optimizer=optimizer)

    def fit(self, model_prefix="model-1"):
        print("Training......")
        callbacks = []
        if model_prefix is not None:
            file_path = DATA_DIR + "cache/%s-{epoch:03d}-{val_loss:.4f}.hdf5" % model_prefix
            callbacks.append(
                ModelCheckpoint(file_path, monitor='val_loss', mode='min', save_weights_only=True, verbose=1))

        RNNModel.start_new_session()
        print("Session created")

        print("Starting Training.....")
        history = self.model.fit(
            self.X_train, self.Y_train,
            epochs=self.n_epochs, batch_size=self.batch_size, validation_data=(self.X_valid, self.Y_valid),
            callbacks=callbacks)
        print("Training Completed")

        if model_prefix is not None:
            file_path = DATA_DIR + 'cache/%s-history.pickle' % model_prefix
            with open(file_path, 'wb') as handle:
                pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Training Completed")
