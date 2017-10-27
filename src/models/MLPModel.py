import pickle

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Merge
from keras.layers.core import Reshape, Activation, Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.utils.generic_utils import get_custom_objects

from common.utils import haversine
from models import Model


class MLPModel(Model):
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
        with open("cache/model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights("cache/model.h5")
        print("Saved model to disk")

    def load(self, model_prefix='model'):
        json_file = open('cache/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("cache/model.h5")
        print("Loaded model from disk")

        return loaded_model

    def create_model(self):
        embed_quarter_hour = Sequential()
        embed_quarter_hour.add(Embedding(self.metadata['n_quarter_hours'], self.embedding_dim, input_length=1))
        embed_quarter_hour.add(Reshape((self.embedding_dim,)))

        embed_day_of_week = Sequential()
        embed_day_of_week.add(Embedding(self.metadata['n_days_per_week'], self.embedding_dim, input_length=1))
        embed_day_of_week.add(Reshape((self.embedding_dim,)))

        embed_week_of_year = Sequential()
        embed_week_of_year.add(Embedding(self.metadata['n_weeks_per_year'], self.embedding_dim, input_length=1))
        embed_week_of_year.add(Reshape((self.embedding_dim,)))

        embed_client_ids = Sequential()
        embed_client_ids.add(Embedding(self.metadata['n_client_ids'], self.embedding_dim, input_length=1))
        embed_client_ids.add(Reshape((self.embedding_dim,)))

        embed_taxi_ids = Sequential()
        embed_taxi_ids.add(Embedding(self.metadata['n_taxi_ids'], self.embedding_dim, input_length=1))
        embed_taxi_ids.add(Reshape((self.embedding_dim,)))

        embed_stand_ids = Sequential()
        embed_stand_ids.add(Embedding(self.metadata['n_stand_ids'], self.embedding_dim, input_length=1))
        embed_stand_ids.add(Reshape((self.embedding_dim,)))

        coords = Sequential()
        coords.add(Dense(1, kernel_initializer="normal", input_dim=20))

        feature_list = [embed_quarter_hour, embed_day_of_week, embed_week_of_year, embed_client_ids, embed_taxi_ids,
                        embed_stand_ids, coords]
        features = Merge(feature_list, mode='concat')

        self.model = Sequential()
        self.model.add(features)

        self.model.add(Dense(500))
        self.model.add(Activation('relu'))

        self.model.add(Dense(len(self.clusters)))
        self.model.add(Activation('softmax'))
        cast_clusters = K.cast_to_floatx(self.clusters)

        def destination(probabilities):
            return tf.matmul(probabilities, cast_clusters)

        get_custom_objects().update({'destination': Activation(destination)})

        self.model.add(Activation(destination))
        optimizer = SGD(lr=0.01, momentum=0.9, clipvalue=1.)
        self.model.compile(loss=haversine(tensor=True), optimizer=optimizer)

    def fit(self, model_prefix="model-1"):
        print("Training......")
        callbacks = []
        if model_prefix is not None:
            file_path = "cache/%s-{epoch:03d}-{val_loss:.4f}.hdf5" % model_prefix
            callbacks.append(
                ModelCheckpoint(file_path, monitor='val_loss', mode='min', save_weights_only=True, verbose=1))

        MLPModel.start_new_session()
        print("Session created")

        print("Starting Training.....")
        history = self.model.fit(
            self.X_train, self.Y_train,
            epochs=self.n_epochs, batch_size=self.batch_size, validation_data=(self.X_valid, self.Y_valid),
            callbacks=callbacks)
        print("Training Completed")

        if model_prefix is not None:
            file_path = 'cache/%s-history.pickle' % model_prefix
            with open(file_path, 'wb') as handle:
                pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Training Completed")
