import importlib
import logging
import random
import string
from time import strftime
import sys
import os
from os.path import dirname, abspath

formatter = logging.Formatter(
    fmt='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S'
)


# from stackoverflow.com/a/11233293/2474159
def setup_logger(name, log_file, level):
    # delay=True creates the logfile with the first line, so empty logfiles won't be created
    # via stackoverflow.com/a/19656056/2474159
    handler = logging.FileHandler(log_file, delay=True)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def get_unique_file_logger(cls, level=logging.INFO):
    rand_str = ''
    for i in range(0, 5):
        rand_str += random.choice(string.ascii_lowercase)
    log_file = cls.__class__.__name__ + '_' + strftime('%d-%m-%Y_%H-%M-%S') + '_' + rand_str + '.log'
    if not in_pyinstaller_mode():  # else is GTPengine or executable, in that case we can't expect a folder
        project_root_dir = dirname(dirname(abspath(__file__)))
        log_file = os.path.join(os.path.join(project_root_dir, 'logs'), log_file)
    logger = setup_logger(rand_str, log_file, level)
    logger.propagate = False  # via stackoverflow.com/a/2267567/2474159
    return logger


# sys._MEIPASS is the path to a temporary folder pyinstaller (re)creates
# therefore the existence of this attribute means we are running from pyinstaller
def in_pyinstaller_mode():
    return hasattr(sys, '_MEIPASS')


# ported to Python 3 from stackoverflow.com/a/44446822
def set_keras_backend(backend):
    from keras import backend as K
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend
