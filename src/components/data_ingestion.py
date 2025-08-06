from src.logger import config_logger
from src.exception import MyException
import pandas as pd
from src.constants.data_ingestion_constants import *
from src.constants  import *
from src.entity.data_ingestion_artifact import DataIngestionConfig
import sys

logger = config_logger('data_ingestion')

class DataIngestion:
    '''loads raw data from path and save in the  artifact dir'''

    def __init__(self,data_ingestion_config: DataIngestionConfig):
        ''' initialize the DataIngestionConfig  with config 
            
            args:
                data_ingestion_config: DataIngestionConfig Dataclass
                '''
        self.data_ingestion_config = data_ingestion_config


    def load_and_save_data(self,input_path:str,output_path:str):
        '''loads the data from input path and save in the artifact dir'''
        try:
            dataset = pd.read_csv(input_path)
            dataset.to_csv(output_path,index=False)
            del dataset #release the dataframe from memory


            logger.info(f'dataset saved here:{output_path} successfully')
        except Exception as e:
            raise MyException(e,sys,logger)
            
    def load_all(self):
        ''' it will load all the dataset from the RAW_DATA_URL And save in the ARTIFACT_RAW_DIR'''
        try:
            
            self.load_and_save_data(self.data_ingestion_config.application_data,
                                    self.data_ingestion_config.save_application_data)
            
            self.load_and_save_data(self.data_ingestion_config.bureau_data,
                                    self.data_ingestion_config.save_bureau_data)
            
            self.load_and_save_data(self.data_ingestion_config.bureau_balance_data,
                                    self.data_ingestion_config.save_bureau_balance_data)
            
            self.load_and_save_data(self.data_ingestion_config.credit_card_balance,
                                    self.data_ingestion_config.save_credit_card_balance)
            
            self.load_and_save_data(self.data_ingestion_config.installment_payments_data,
                                    self.data_ingestion_config.save_installment_payments_data)
            
            self.load_and_save_data(self.data_ingestion_config.pos_cash_data,
                                    self.data_ingestion_config.save_pos_cash_data)
            
            self.load_and_save_data(self.data_ingestion_config.previous_application_data,
                                    self.data_ingestion_config.save_previous_application_data)
            logger.info(f'all dataset loaded and saved in {ARTIFACT_RAW_DIR} successfully')
        except Exception as e:
            raise MyException(e,sys,logger)

if __name__ == '__main__':
    try:    
        ingestion = DataIngestion(DataIngestionConfig())
        ingestion.load_all()
    except Exception as e:
        logger.error(f'error occured during the data_ingestion{e}')
        raise e

