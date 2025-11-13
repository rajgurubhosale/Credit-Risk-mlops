from src.utils.main_utils import *
from src.constants import *
from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file
from src.constants.data_validaton_constant import *

@dataclass
class DataValidationConfig:
    logger: object = any
    schema_config_file_path:str = SCHEMA_CONFIG_FILE_PATH
    

    def load_dataset_schema_mapping(self):
        '''loads the dataset schema mapping file '''
        self.logger.info('pass')
        try:
            file =  read_yaml_file(self.schema_config_file_path,self.logger)
            return file.get('datasets')
        
        except Exception as e:
            raise MyException(e,sys,self.logger)
