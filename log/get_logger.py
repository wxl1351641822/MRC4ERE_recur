import os
import sys
import logging
from datetime import datetime

sys.path.append("..")
def get_logger(id,text=''):
    logger = logging.getLogger(__name__)

    logger.setLevel(level = logging.INFO)
    log_path = "../log/log_output_vote_token"
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    handler = logging.FileHandler(os.path.join(log_path, "{}log-{}".format(text,id)))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%Y%b%d-%H:%M:%S")
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger



