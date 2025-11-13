from src.utils.main_utils import read_yaml_file
import pandas as pd
import sys
from src.logger import config_logger
from src.entity.data_validation_artifact import *
import os


logger = config_logger('demo')
logger.info('pass')
dv = DataValidationConfig(logger)
file = dv.load_dataset_schema_mapping()
print(file)