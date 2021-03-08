import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from config import Config

# Logging tools in this file are a modified version of this post: https://www.toptal.com/python/in-depth-python-logging


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(Config.LOG_FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(Config.LOG_FILE, when='midnight')
    file_handler.setFormatter(Config.LOG_FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(Config.LOG_LEVEL)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger
