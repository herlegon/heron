import logging

# logging.basicConfig(level=logging.CRITICAL, format='%(filename)s (%(lineno)d): %(funcName)s: %(message)s')
hlogger: logging.Logger = logging.getLogger("heron_logger")

# Enable this to debug imports
import sys
hlogger.addHandler(logging.StreamHandler(sys.stdout))
hlogger.setLevel(logging.WARNING)
