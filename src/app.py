import getopt
import sys

from models import MLPModel
from preprocessor.clustering import MeanShiftClustering
from preprocessor.loader import Data


def arg_parser():
    params = dict()
    myopts, args = getopt.getopt(sys.argv[1:], "t:m:")
    params['is_train'] = False

    for o, a in myopts:
        if o == '-m':
            params['model_name'] = a
        elif o == '-t':
            params['is_train'] = True if a=='train' else False
        else:
            print("Usage: %s -m input -o output" % sys.argv[0])

    return params


if __name__ == '__main__':
    params = arg_parser()
    model_name = params['model_name']
    if params['is_train']:
        print(" DESTINATION PREDICTION : Training model using {} : %s".format(model_name))

        data = Data()
        data.load_data()

        ms = MeanShiftClustering(data.Y_train)
        ms.fit()
        clusters = ms.get_cluster_centroids()

        mlp = MLPModel.MLPModel()
        mlp.set_data(data)
        mlp.set_clusters(clusters)
        mlp.create_model()
        mlp.fit()
        mlp.save()
    else:
        print('Prediction not implemented yet')
