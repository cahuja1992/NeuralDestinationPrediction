import abc


class Model(object):
    """
    Abstract Class for Model
    """
    __metadata__ = abc.ABCMeta

    @abc.abstractmethod
    def create_model(self):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self, features):
        pass

    @abc.abstractmethod
    def save(self, location):
        pass

    @abc.abstractmethod
    def load(self, location):
        pass
