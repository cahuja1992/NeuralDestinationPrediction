import abc

import pandas as pd
import sklearn


class Clusters(object):
    __metadata__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def get_cluster_centroids(self):
        pass


class MeanShiftClustering(Clusters):
    def __init__(self, coords):
        self.coords = coords
        self.clustering_model = None

    def fit(self):
        clusters = pd.DataFrame({
            'approx_latitudes': self.coords[:, 0].round(4),
            'approx_longitudes': self.coords[:, 1].round(4)
        })
        clusters = clusters.drop_duplicates(['approx_latitudes', 'approx_longitudes'])
        clusters = clusters.as_matrix()
        bandwidth = sklearn.cluster.estimate_bandwidth(clusters, quantile=0.0002)
        self.clustering_model = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        self.clustering_model.fit(clusters)

    def get_cluster_centroids(self):
        return self.clustering_model.cluster_centers_
