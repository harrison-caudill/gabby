import datetime
import gc
import lmdb
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pprint
import subprocess
import struct
import sys
import tletools
import scipy
import scipy.interpolate
import scipy.signal

from .defs import *
from .utils import *

"""
"""


class Moses(object):
    """Forward propagator.

    
    """

    def __init__(self, cfg, moral_decay, ):
        self.cfg = cfg
        self.db_env = env
        self.db_apt = apt
        self.db_tle = tle
        self.db_scope = scope
