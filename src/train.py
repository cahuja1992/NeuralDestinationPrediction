from models import MLPModel
from preprocessor.clustering import MeanShiftClustering
from preprocessor.loader import Data

if __name__ == '__main__':
    data = Data()
    data.load_data()

    ms = MeanShiftClustering()
    ms.fit()
    clusters = ms.get_cluster_centroids()

    mlp = MLPModel.MLPModel()
    mlp.set_data(data)
    mlp.create_model()
    mlp.fit()
    mlp.save()
