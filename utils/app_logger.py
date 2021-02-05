import logging
from logging.handlers import RotatingFileHandler
import os

from utils.file_utils import read_json

config_path = os.path.join(os.getcwd(), 'configuration.json')
config_data = read_json(config_path)


class AppLogger:
    __instance = None

    @staticmethod
    def log_setup():
        log_handler = logging.handlers.WatchedFileHandler(config_data["log_path"])
        formatter = logging.Formatter('%(asctime)s [%(process)d]: %(message)s', '%b %d %H:%M:%S')
        log_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(console_handler)
        logger.addHandler(log_handler)
        logger.setLevel(logging.DEBUG)
        return logger

    @staticmethod
    def getInstance():
        """ Static access method. """
        if AppLogger.__instance is None:
            logger = AppLogger.log_setup()
            return logger

        return AppLogger.__instance

    def __init__(self):
        self.input_dict = {}
        """ Virtually private constructor. """
        if AppLogger.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            AppLogger.__instance = self
