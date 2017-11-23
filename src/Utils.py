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
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=5))
    log_file = cls.__class__.__name__ + '_' + strftime('%d-%m-%Y_%H-%M-%S') + '_' + rand_str + '.log'
    if running_run_script():  # else is GTPengine or executable, in that case we can't expect a folder
        project_dir = dirname(dirname(abspath(__file__)))
        log_file = os.path.join(os.path.join(project_dir, 'logs'), log_file)
    logger = setup_logger(rand_str, log_file, level)
    logger.propagate = False  # via stackoverflow.com/a/2267567/2474159
    return logger


# true if the current run was started using run.py
# false if not, that's the case e.g. when running GTPengine directly and when building an executable of GTPengine
# better name for this method?
def running_run_script():
    return str(sys.argv[0]).endswith('run.py')
