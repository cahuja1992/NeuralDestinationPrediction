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

from common.utils import haversine_tf
from models import Model
from config import DATA_DIR
from common.logging import LOG
import config
import numpy as np
from sklearn.preprocessing import scale


class MLPModel(Model):
    def __init__(self):
        self.metadata = None
        self.clusters = None
        self.lr = config.lr
        self.momentum = config.momentum
        self.clip_value = config.clip_value
        self.model = None
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs
        self.X_train, self.Y_train = None, None
        self.X_valid, self.Y_valid = None, None
        self.X_test, self.Y_test = None, None

    def process_features(self, df):
        def first_last_k(coords):
            try:
                k = 5
                partial = [coords[0] for i in range(2 * k)]
                num_coords = len(coords)
                if num_coords < 2 * k:
                    partial[-num_coords:] = coords
                else:
                    partial[:k] = coords[:k]
                    partial[-k:] = coords[-k:]
                partial = np.row_stack(partial)
                return np.array(partial).flatten()
            except:
                LOG.debug(type(coords))

        print("Processing features.....")
        coords = np.row_stack(df['POLYLINE'].apply(first_last_k))
        latitudes = coords[:, ::2]
        coords[:, ::2] = scale(latitudes)
        longitudes = coords[:, 1::2]
        coords[:, 1::2] = scale(longitudes)

        return [
            df['QUARTER_HOUR'].as_matrix(),
            df['DAY_OF_WEEK'].as_matrix(),
            df['WEEK_OF_YEAR'].as_matrix(),
            df['ORIGIN_CALL_ENCODED'].as_matrix(),
            df['TAXI_ID_ENCODED'].as_matrix(),
            df['ORIGIN_STAND_ENCODED'].as_matrix(),
            coords,
        ]

    def set_data(self, data):
        self.X_train, self.Y_train = MLPModel.process_features(data.X_train), data.Y_train
        self.X_valid, self.Y_valid = MLPModel.process_features(data.X_valid), data.Y_valid
        self.X_test, self.Y_test = MLPModel.process_features(data.X_test), data.Y_test
        self.metadata = data.get_metadata

    def set_clusters(self, clusters):
        self.clusters = clusters

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
        with open(DATA_DIR + "model/model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(DATA_DIR + "model/model.h5")
        LOG.info("Saved model to disk")

    def load(self, model_prefix='latest'):
        json_file = open(DATA_DIR + 'model/{}-model.json'.format(model_prefix), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(DATA_DIR + "model/{}-model.h5".format(model_prefix))
        LOG.info("Loaded model from disk")

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
        self.model.compile(loss=haversine_tf, optimizer=optimizer)

    def fit(self, model_prefix="model-1"):
        LOG.info("Training......")
        callbacks = []
        if model_prefix is not None:
            file_path = DATA_DIR + "model/%s-{epoch:03d}-{val_loss:.4f}.hdf5" % model_prefix
            callbacks.append(
                ModelCheckpoint(file_path, monitor='val_loss', mode='min', save_weights_only=True, verbose=1))

        MLPModel.start_new_session()
        LOG.info("Session created")

        LOG.info("Starting Training.....")
        history = self.model.fit(
            self.X_train, self.Y_train,
            epochs=self.n_epochs, batch_size=self.batch_size, validation_data=(self.X_valid, self.Y_valid),
            callbacks=callbacks)
        LOG.info("Training Completed")

        if model_prefix is not None:
            file_path = DATA_DIR + 'model/%s-history.pickle' % model_prefix
            with open(file_path, 'wb') as handle:
                pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        LOG.info("Training Completed")
