import astropy
import datetime
import lmdb
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pprint
import struct
import sys

from .defs import *


def mkdir_p(path):
    """Emulates the functionality of mkdir -p
    """
    if not len(path): return
    if not os.path.isdir(path):
        logging.debug('Creating Directory: %s' % path)
        os.makedirs(path)

def setup_logging(log_file='output/gab.log', log_level='info'):
    """Gets the logging infrastructure up and running.

    log_file: either relative or absolute path to the log file.
    log_level: <str> representing the log level
    """

    # Let's make sure it has a directory and that we can write to that
    # file.
    log_dir = os.path.dirname(log_file)
    if not len(log_dir): log_dir = '.'
    if not os.path.exists(log_dir): mkdir_p(os.path.dirname(log_file))

    if not os.path.isdir(log_dir):
        msg = ("Specified logging directory is " +
               "not a directory: %s" % log_dir)
        logging.critical(msg)
        raise ConfigurationException(msg)

    try:
        fd = open(log_file, 'w')
    except:
        msg = "Failed to open log file for writing: %s" % log_file
        logging.critical(msg)
        raise ConfigurationException(msg)

    if log_level.lower() not in ['critical',
                                 'error',
                                 'warning',
                                 'info',
                                 'debug']:
        msg = "Invalid log level: %s" % log_level
        logging.critical(msg)
        raise ConfigurationException(msg)
    log_level = log_level.upper()

    # Add the log file
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setStream(fd)
    logging.getLogger().addHandler(ch)

    logging.getLogger().setLevel(log_level)

    # Add stdout
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setStream(sys.stdout)
    logging.getLogger().addHandler(ch)


def time_command(f, msg):
    def retval(*args, **kwargs):
        logging.info("%s"%msg)
        start = datetime.datetime.now().timestamp()
        f(*args, **kwargs)
        end = datetime.datetime.now().timestamp()
        ms = int((end - start)*1000)
        logging.info("  Call completed in {ms}ms")

def plot_apt(frag, tapt, path):
    fig = plt.figure(figsize=(12, 8))
    fig.set_dpi(300)
    if frag: fig.suptitle(f"Decay profile: {frag}")
    else: fig.suptitle(f"Decay profile")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Time Since Launch (days)')
    ax.set_ylabel('Orbital Altitude (km)')

    t, A, P, T = tapt
    t -= t[0]
    plt_A = ax.plot(t, A, color='firebrick', label='Apogee')
    plt_P = ax.plot(t, P, color='dodgerblue', label='Perigee')
    ax.legend(loc=1)

    ax = ax.twinx()
    plt_T = ax.plot(t, T/60.0, color='black', label='Period')
    ax.set_ylabel('Period (minutes)')
    ax.legend(loc=2)

    plts = plt_A + plt_P + plt_T
    labs = [p.get_label() for p in plts]
    ax.legend(plts, labs, loc=1)

    fig.savefig(path)
