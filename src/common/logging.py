import logging
import os

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

fh = logging.FileHandler(os.getcwd() + "application.log")

# create formatter
formatter = logging.Formatter("%(asctime)s;%(levelname)s;(%(threadName)-10s); ""%(message)s")

# add formatter to handler
ch.setFormatter(formatter)
fh.setFormatter(formatter)

LOG.addHandler(ch)
LOG.addHandler(fh)
