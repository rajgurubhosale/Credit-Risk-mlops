from src.logger import config_logger
from src.exception import MyException
from src.constants.artifacts_paths import *
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
import sys  
import shutil
logger = config_logger('module_01_data_ingestion.py')

class DataIngestion:
    '''loads raw data from path and save in the  artifact dir'''

    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig,
        data_ingestion_artifact:DataIngestionArtifact
        ):
        ''' initialize the DataIngestionConfig  with config 
            
            args:
                data_ingestion_config: DataIngestionConfig Dataclass
                '''
        self.data_ingestion_config = data_ingestion_config
        self.data_ingestion_artifact = data_ingestion_artifact
        # to save the artifacts
        self.data_ingestion_artifact.artifact_data_raw_dir.mkdir(
                parents=True,
                exist_ok=True
            )
        
       

    def load_and_save_data(self,input_filename:str):
        '''loads the data from RAW_DATA_URL\input_filename
            And Saves data in artifact\raw 
            args:
                input_filename: fileame of the dataset in The RAW_DATA_URL
        '''
        try:
            #load data from this dir
            # source file
            source_file_path = self.data_ingestion_config.source_raw_data_url / input_filename
            
            if not source_file_path.exists():
                raise FileNotFoundError(
                    f"{input_filename} not found in "
                    f"{self.data_ingestion_config.source_raw_data_url}"
                )

            output_path = self.data_ingestion_artifact.artifact_data_raw_dir / input_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source_file_path, output_path)
            
            logger.info(f'dataset saved here:{output_path} successfully')
        except Exception as e:
            raise MyException(e,sys,logger)
            

    def load_all(self):
        ''' loads all the dataset from the RAW_DATA_URL And save in the artifact_data_raw_dir'''
        try:
        
            file_list = self.data_ingestion_config.source_data_names_list

            for filename in file_list:
                self.load_and_save_data(filename)

            logger.info(f'all dataset loaded and saved in {self.data_ingestion_artifact.artifact_data_raw_dir} successfully')
        except Exception as e:
            raise MyException(e,sys,logger)

if __name__ == '__main__':
    try:    
        ingestion = DataIngestion(DataIngestionConfig(),DataIngestionArtifact())
        ingestion.load_all()
    except Exception as e:
        logger.error(f'error occured during the data_ingestion{e}')
        raise e

