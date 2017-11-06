import pickle

import tensorflow as tf
from keras import backend as k
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD
from keras.utils.generic_utils import get_custom_objects
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Dense, LSTM, Embedding, Reshape, Merge

from common.utils import haversine_tf
from models import Model
import config
import numpy as np
from common.logging import LOG


class BidirectionalRNN(Model):
    def __init__(self):
        self.clusters = None
        self.lr = config.lr
        self.momentum = config.momentum
        self.clip_value = config.clip_value
        self.model = None
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs

        self.X_train, self.Y_train = None, None
        self.X_valid, self.Y_valid = None, None
        self.X_test, self.Y_test = None, None
        self.metadata = None
        self.time_step = None
        self.num_features = None

    def process_features(self, df, n_steps=20):
        coords = df['POLYLINE'].apply(lambda x: np.array(x).flatten()).values.tolist()
        coords = pad_sequences(coords, maxlen=n_steps * 2, dtype='float32')
        coords = np.reshape(coords, [coords.shape[0], n_steps, 2])
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
        self.X_train, self.Y_train = self.process_features(data.X_train), data.Y_train
        self.X_valid, self.Y_valid = self.process_features(data.X_valid), data.Y_valid
        self.X_test, self.Y_test = self.process_features(data.X_test), data.Y_test

        self.time_step = self.X_train[-1].shape[1]
        self.num_features = self.X_train[-1].shape[2]
        self.metadata = data.get_metadata

    def set_clusters(self, clusters):
        self.clusters = clusters

    @staticmethod
    def start_new_session():
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True

        session = tf.Session(config=conf, graph=tf.get_default_graph())
        k.tensorflow_backend.set_session(session)

    def predict(self, features):
        self.model.predict(features)

    def save(self, model_prefix='model'):
        model_json = self.model.to_json()
        with open(config.DATA_DIR + "model/model.json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(config.DATA_DIR + "model/model.h5")
        LOG.info("Saved model to disk")

    def load(self, model_prefix='latest'):
        json_file = open(config.DATA_DIR + 'model/{}-model.json'.format(model_prefix), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(config.DATA_DIR + "model/{}-model.h5".format(model_prefix))
        LOG.info("Loaded model from disk")

        return loaded_model

    def create_model(self):
        embedding_dim = 10

        embed_quarter_hour = Sequential()
        embed_quarter_hour.add(Embedding(self.metadata['n_quarter_hours'], embedding_dim, input_length=1))
        embed_quarter_hour.add(Reshape((embedding_dim,)))

        embed_day_of_week = Sequential()
        embed_day_of_week.add(Embedding(self.metadata['n_days_per_week'], embedding_dim, input_length=1))
        embed_day_of_week.add(Reshape((embedding_dim,)))

        embed_week_of_year = Sequential()
        embed_week_of_year.add(Embedding(self.metadata['n_weeks_per_year'], embedding_dim, input_length=1))
        embed_week_of_year.add(Reshape((embedding_dim,)))

        embed_client_ids = Sequential()
        embed_client_ids.add(Embedding(self.metadata['n_client_ids'], embedding_dim, input_length=1))
        embed_client_ids.add(Reshape((embedding_dim,)))

        embed_taxi_ids = Sequential()
        embed_taxi_ids.add(Embedding(self.metadata['n_taxi_ids'], embedding_dim, input_length=1))
        embed_taxi_ids.add(Reshape((embedding_dim,)))

        embed_stand_ids = Sequential()
        embed_stand_ids.add(Embedding(self.metadata['n_stand_ids'], embedding_dim, input_length=1))
        embed_stand_ids.add(Reshape((embedding_dim,)))

        coords = Sequential()
        coords.add(LSTM(100, input_shape=(self.time_step, 2),
                        unit_forget_bias=True,
                        activation='tanh',
                        dropout=0.2,
                        recurrent_initializer='glorot_uniform',
                        recurrent_activation='sigmoid',
                        recurrent_dropout=0.2))

        features_list = [embed_quarter_hour, embed_day_of_week, embed_week_of_year, embed_client_ids, embed_taxi_ids,
                         embed_stand_ids, coords]
        features = Merge(features_list, mode='concat')

        self.model = Sequential()
        self.model.add(features)
        self.model.add(Dense(100))
        self.model.add(Activation('relu'))
        self.model.add(Dense(len(self.clusters)))
        self.model.add(Activation('softmax'))

        cast_clusters = k.cast_to_floatx(self.clusters)

        def distance(probabilities):
            return tf.matmul(probabilities, cast_clusters)

        get_custom_objects().update({'distance': Activation(distance)})
        self.model.add(Activation(distance))

        optimizer = SGD(lr=config.lr, momentum=config.momentum, clipvalue=config.clip_value)
        self.model.compile(loss=haversine_tf, optimizer=optimizer)

    def fit(self, model_prefix="model-1"):
        LOG.info("Training......")
        callbacks = []
        if model_prefix is not None:
            file_path = config.DATA_DIR + "model/%s-{epoch:03d}-{val_loss:.4f}.hdf5" % model_prefix
            callbacks.append(
                ModelCheckpoint(file_path, monitor='val_loss', mode='min', save_weights_only=True, verbose=1))

        RNNModel.start_new_session()
        LOG.info("Session created")

        LOG.info("Starting Training.....")
        history = self.model.fit(
            self.X_train, self.Y_train,
            epochs=self.n_epochs, batch_size=self.batch_size, validation_data=(self.X_valid, self.Y_valid),
            callbacks=callbacks)
        LOG.info("Training Completed")

        if model_prefix is not None:
            file_path = config.DATA_DIR + 'model/%s-history.pickle' % model_prefix
            with open(file_path, 'wb') as handle:
                pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        LOG.info("Training Completed")
