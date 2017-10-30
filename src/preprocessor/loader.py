import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

from common.logging import LOG
from common.utils import str_to_list
from config import DATA_DIR


class Data:
    def __init__(self):
        self.train_cache = DATA_DIR+'cache/train.pickle'
        self.train_labels_cache = DATA_DIR+'cache/train-labels.npy'
        self.validation_cache = DATA_DIR+'cache/validation.pickle'
        self.validation_labels_cache = DATA_DIR+'cache/validation-labels.npy'
        self.test_cache = DATA_DIR+'cache/test.pickle'
        self.test_labels_cache = DATA_DIR+'cache/test-labels.npy'
        self.competition_test_cache = DATA_DIR+'cache/competition-test.pickle'
        self.metadata_cache = DATA_DIR+'cache/metadata.pickle'

        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
        self.Y_test = None
        self.metadata = None

    @staticmethod
    def random_truncate(coords):
        if len(coords) <= 1:
            return coords
        n = np.random.randint(len(coords) - 1)

        if n > 0:
            return coords[:-n]
        else:
            return coords

    @staticmethod
    def encode_feature(feature, train, test):
        encoder = LabelEncoder()
        train_values = train[feature].copy()
        test_values = test[feature].copy()

        train_values[np.isnan(train_values)] = 0
        test_values[np.isnan(test_values)] = 0

        encoder.fit(pd.concat([train_values, test_values]))

        train[feature + '_ENCODED'] = encoder.transform(train_values)
        test[feature + '_ENCODED'] = encoder.transform(test_values)
        return encoder

    @staticmethod
    def extract_features(df):
        df['POLYLINE'] = df['POLYLINE'].apply(str_to_list)
        df['START_LAT'] = df['POLYLINE'].apply(lambda x: x[0][0])
        df['START_LONG'] = df['POLYLINE'].apply(lambda x: x[0][1])

        datetime_index = pd.DatetimeIndex(df['TIMESTAMP'])
        df['QUARTER_HOUR'] = datetime_index.hour * 4 + datetime_index.minute / 15
        df['DAY_OF_WEEK'] = datetime_index.dayofweek
        df['WEEK_OF_YEAR'] = datetime_index.weekofyear - 1
        df['DURATION'] = df['POLYLINE'].apply(lambda x: 15 * len(x))

    @staticmethod
    def remove_outliers(df, labels):
        indices = np.where((df.DURATION > 60) & (df.DURATION <= 2 * 3600))
        df = df.iloc[indices]
        labels = labels[indices]

        bounds = ((41.052431, -8.727951), (41.257678, -8.456039))
        indices = np.where(
            (labels[:, 0] >= bounds[0][0]) &
            (labels[:, 1] >= bounds[0][1]) &
            (labels[:, 0] <= bounds[1][0]) &
            (labels[:, 1] <= bounds[1][1])
        )
        df = df.iloc[indices]
        labels = labels[indices]

        return df, labels

    @staticmethod
    def get_features(df):
        def first_last_k(coordinates):
            k = 5
            partial = [coordinates[0] for i in range(2 * k)]
            num_coords = len(coordinates)
            if num_coords < 2 * k:
                partial[-num_coords:] = coordinates
            else:
                partial[:k] = coordinates[:k]
                partial[-k:] = coordinates[-k:]
            partial = np.row_stack(partial)
            return np.array(partial).flatten()

        LOG.info("Processing features.....")
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

    def load_data(self):
        if os.path.isfile(self.train_cache):
            LOG.info("Found cached data")
            train = pd.read_pickle(self.train_cache)
            validation = pd.read_pickle(self.validation_cache)
            test = pd.read_pickle(self.test_cache)

            train_labels = np.load(self.train_labels_cache)
            validation_labels = np.load(self.validation_labels_cache)
            test_labels = np.load(self.test_labels_cache)

            competition_test = pd.read_pickle(self.competition_test_cache)
            with open(self.metadata_cache, 'rb') as handle:
                metadata = pickle.load(handle)
            LOG.info("Data Loaded")
        else:
            LOG.info("Cached data not found....")
            LOG.info("Tranforming data")
            datasets = []
            for kind in ['train', 'test']:
                csv_file = '{0}/{1}.csv'.format(DATA_DIR, kind)
                df = pd.read_csv(csv_file)
                df = df[0:1000]
		df = df[df['MISSING_DATA'] == False]
                df = df[df['POLYLINE'] != '[]']
                df.drop('MISSING_DATA', axis=1, inplace=True)
                df.drop('DAY_TYPE', axis=1, inplace=True)
                df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64[s]')
                Data.extract_features(df)
                datasets.append(df)

            train, competition_test = datasets

            client_encoder = Data.encode_feature('ORIGIN_CALL', train, competition_test)
            taxi_encoder = Data.encode_feature('TAXI_ID', train, competition_test)
            stand_encoder = Data.encode_feature('ORIGIN_STAND', train, competition_test)

            train['POLYLINE_FULL'] = train['POLYLINE'].copy()
            train['POLYLINE'] = train['POLYLINE'].apply(Data.random_truncate)
            train_labels = np.column_stack([
                train['POLYLINE_FULL'].apply(lambda x: x[-1][0]),
                train['POLYLINE_FULL'].apply(lambda x: x[-1][1])
            ])
            train, train_labels = Data.remove_outliers(train, train_labels)

            metadata = {
                'n_quarter_hours': 96,  # Number of quarter of hours in one day (i.e. 24 * 4).
                'n_days_per_week': 7,
                'n_weeks_per_year': 52,
                'n_client_ids': len(client_encoder.classes_),
                'n_taxi_ids': len(taxi_encoder.classes_),
                'n_stand_ids': len(stand_encoder.classes_),
            }

            train, validation, train_labels, validation_labels = train_test_split(train, train_labels, test_size=0.02)
            validation, test, validation_labels, test_labels = train_test_split(validation, validation_labels,
                                                                                test_size=0.5)

            train.to_pickle(self.train_cache)
            validation.to_pickle(self.validation_cache)
            test.to_pickle(self.test_cache)
            np.save(self.train_labels_cache, train_labels)
            np.save(self.validation_labels_cache, validation_labels)
            np.save(self.test_labels_cache, test_labels)
            competition_test.to_pickle(self.competition_test_cache)
            with open(self.metadata_cache, 'wb') as handle:
                pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

            LOG.info("Data Transformed")

        self.X_train = Data.get_features(train)
        self.Y_train = train_labels
        self.X_valid = Data.get_features(validation)
        self.Y_valid = validation_labels
        self.X_test = Data.get_features(test)
        self.Y_test = Data.get_features(test_labels)
        self.metadata = metadata
