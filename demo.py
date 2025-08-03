from src.exception import MyException
import sys
from src.logger import config_logger

logger = config_logger('check_exception')

try:
    print(1+'z')
except Exception as e:
    raise MyException(e,sys,logger) from e
