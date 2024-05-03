import logging
import sys
import warnings
import os


def configure_logging_libs(debug=False):
    modules = [
      "numba",
      "httpx",
      "markdown_it",
      "fairseq",
      "faiss",
    ]
    try:
        for module in modules:
            logging.getLogger(module).setLevel(logging.WARNING)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" if not debug else "1"

    except Exception as error:
        logger.error(str(error))


def setup_logger(name_log):
    logger = logging.getLogger(name_log)
    logger.setLevel(logging.INFO)

    _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
    _default_handler.flush = sys.stderr.flush
    logger.addHandler(_default_handler)

    logger.propagate = False

    handlers = logger.handlers

    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s] >> %(message)s")
        handler.setFormatter(formatter)

    # logger.handlers

    configure_logging_libs()

    return logger


logger = setup_logger("infer_rvc_python")
logger.setLevel(logging.INFO)
