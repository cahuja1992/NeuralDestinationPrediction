import json

import numpy as np
import tensorflow as tf


def str_to_list(string):
    return [(lat, long) for (long, lat) in json.loads(string)]


def haversine(lat_lon_1, lat_lon_2, tensor=False):
    lat1 = lat_lon_1[:, 0]
    lon1 = lat_lon_1[:, 1]
    lat2 = lat_lon_2[:, 0]
    lon2 = lat_lon_2[:, 1]

    r_earth = 6371
    if not tensor:
        lat = np.abs(lat1 - lat2) * np.pi / 180
        lon = np.abs(lon1 - lon2) * np.pi / 180
        lat1 = lat1 * np.pi / 180
        lat2 = lat2 * np.pi / 180
        a = np.sin(lat / 2) * np.sin(lat / 2) + np.cos(lat1) * np.cos(lat2) * np.sin(lon / 2) * np.sin(lon / 2)
        d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return r_earth * d
    elif tensor:
        lat = tf.abs(lat1 - lat2) * np.pi / 180
        lon = tf.abs(lon1 - lon2) * np.pi / 180
        lat1 = lat1 * np.pi / 180
        lat2 = lat2 * np.pi / 180
        a = tf.sin(lat / 2) * tf.sin(lat / 2) + tf.cos(lat1) * tf.cos(lat2) * tf.sin(lon / 2) * tf.sin(lon / 2)
        d = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))
        return r_earth * d
