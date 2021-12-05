import os
import logging

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
    if not os.path.exists(log_dir):
        chirp.mkdir_p(os.path.dirname(log_file))
    if not os.path.isdir(log_dir):
        msg = ("Specified logging directory is " +
               "not a directory: %s" % log_dir)
        logging.critical(msg)
        raise ConfigurationException(msg)
    try:
        fd = open(log_file, 'w')
        fd.close()
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

    # Set up the logging facility
    logging.basicConfig(level=log_level, filename=log_file, filemode='w')
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logging.getLogger().addHandler(ch)
