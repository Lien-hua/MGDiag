import logging
import os
import sys

from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Write console logs without breaking active tqdm progress bars."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)

def get_logger(save_dir, task_name):
    os.makedirs(f"{save_dir}", exist_ok=True)

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_file_name = f'{task_name}.log'
    file_handler = logging.FileHandler(os.path.join(save_dir, log_file_name), mode='w+')
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
