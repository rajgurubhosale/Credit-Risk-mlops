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
        self.artifact_raw_dir = self.data_ingestion_config.artifact_raw_dir
        
        #create artifact\raw dir 
       

    def load_and_save_data(self,input_filename:str):
        '''loads the data from RAW_DATA_URL\input_filename
            And Saves data in artifact\raw 
            args:
                input_filename: fileame of the dataset in The RAW_DATA_URL
        '''
        try:

            #load data from this dir
            source_file_path = os.path.join(self.data_ingestion_config.source_raw_path,input_filename)

            output_path = os.path.join(self.artifact_raw_dir,input_filename)

            dataset = pd.read_csv(source_file_path)
            dataset.to_csv(output_path,index=False)
            
            logger.info(f'dataset saved here:{output_path} successfully')
        except Exception as e:
            raise MyException(e,sys,logger)
            

    def load_all(self):
        ''' loads all the dataset from the RAW_DATA_URL And save in the artifact_raw_dir'''
        try:
            
            file_list = self.data_ingestion_config.load_all_dataset()

            for filename in file_list:
                self.load_and_save_data(filename)

            logger.info(f'all dataset loaded and saved in {self.artifact_raw_dir} successfully')
        except Exception as e:
            raise MyException(e,sys,logger)

if __name__ == '__main__':
    try:    
        ingestion = DataIngestion(DataIngestionConfig())
        ingestion.load_all()
    except Exception as e:
        logger.error(f'error occured during the data_ingestion{e}')
        raise e

