import logging
from importlib import reload
from utils import logging as log_utils


def test_configure_logging_sets_level():
    reload(log_utils)
    log_utils.configure_logging(level=logging.INFO)
    assert logging.getLogger().level == logging.INFO

