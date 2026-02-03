import pandas as pd
import sys
import os
import yaml

from src.logger import config_logger
from  src.utils.main_utils import *

from src.entity.data_validation_artifact import *
from src.entity.data_ingestion_artifact import *

from src.constants import *
from src.constants.data_validaton_constant import *
from src.constants.data_ingestion_constants  import *


logger = config_logger('02_data_validation')

class DataValidation:
    ''' validate the dataset compatiable with  defined cdata schema for  model pipeline
    
        this class ensures that the data loaded in raw sources has the  validate number of columns
        and have expected DataTypes.
    '''

    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_config:DataIngestionConfig):

        ''' initilaize the data_validation_config,data_ingestion_config.    

            args:
                data_validation_config: DataValidationConfig Dataclass contains dataset_schema_mappings
                data_ingestion_config: DataIngestionConfig dataclass containing artifact\dir datapath
        ''' 
        self.data_validation_config = data_validation_config
        self.data_ingestion_config = data_ingestion_config

    def check_dataframe_schema(self,df: pd.DataFrame,yaml_schema_file:dict):

        ''' validates if DataFrame match expected schema such as 
                - columns count mismatch,
                - extra feature
                - missing feature 
            args:
                df: DataFrame to be validated
                yaml_schema_file: yaml file containing exprected schema
                '''
        try:
            file_columns = yaml_schema_file['columns']
            schema_col_set = set()

            for i in file_columns:
                schema_col_set.update((i.keys()))
            df_columns = set(df.columns)
            
            missing_columns_df = 0
            extra_columns_df = 0

            missing_columns_df = schema_col_set.difference(df_columns)
            extra_columns_df = df_columns.difference(schema_col_set)

            if len(file_columns)  != df.shape[1]:
                self.status = False
                self.error_msg += f'expected {file_columns} no of columns. got {df.shape[1]} \n'

            if len(missing_columns_df) > 0:
                self.status = False
                self.error_msg += f'this feature:{ list(missing_columns_df)} are missing in DataFrame\n'

            if len(extra_columns_df) > 0:
                self.status = False
                self.error_msg += f'this columns are extra in the df:{list(extra_columns_df)} \n'


        except Exception as e:
            raise MyException(e,sys,logger)

    def _ensure_instance_yamlfile(self,yaml_schema_columns:dict):
        ''' ensures the yaml schema returns columns always as a  list '''
        
        if isinstance(yaml_schema_columns,list):
            return yaml_schema_columns
        elif isinstance(yaml_schema_columns,str):
            return [yaml_schema_columns]
        elif yaml_schema_columns is None:
            return []
        else:
            return []

    def validate_columns_dtypes(self,df:pd.DataFrame,yaml_schema_file):
        ''' checks if the  categorical and numerical column  match expected schema

            args:
                df: DataFrame to be validated
                yaml_schema_file: yaml file containing expected schema
        '''
        try:

            cat_columns_df = df.select_dtypes(include=['object','category','bool']).columns.tolist()
            num_columns_df = df.select_dtypes(exclude=['object','category','bool']).columns.tolist()
            
            #all columns for the DataFrame from that DataFrame schema
            categorical_columns_schema = self._ensure_instance_yamlfile( yaml_schema_file['categorical_columns'])
            numerical_columns_schema = self._ensure_instance_yamlfile(yaml_schema_file['numerical_columns'])
            all_columns_schema = categorical_columns_schema + numerical_columns_schema

            missing_categorical_columns_df = []
            missing_numerical_columns_df = []

            for column in all_columns_schema:

                if column in categorical_columns_schema:
                    if column not in cat_columns_df:
                        missing_categorical_columns_df.append(column)
                
                elif column in numerical_columns_schema:
                    if column not in num_columns_df:
                        missing_numerical_columns_df.append(column)

            if len(categorical_columns_schema) != 0:
                if len(missing_categorical_columns_df) != 0:
                    self.status = False
                    self.error_msg += f'this Categorical feature are missing in DataFrame:{missing_categorical_columns_df} \n'
            else:
                pass        

            if len(missing_numerical_columns_df) > 0:
                self.status = False
                self.error_msg += f'this Numerical feature are missing in DataFrame:{missing_numerical_columns_df} \n'

        except Exception as e:
            raise MyException(e,sys,logger)
        
    def _verify_artifact_data(self):
        '''verify if data files are present in the artifact_raw_dir or not'''
        #data is  in data ingestion
        try:    
            artifact_raw_dir = self.data_ingestion_config.artifact_raw_dir
        
            if os.listdir(artifact_raw_dir):
                logger.info(f'dataset are present in  {artifact_raw_dir}')
            else:
                logger.error(f'the {artifact_raw_dir} dir is empty')

        except Exception as e:
            raise MyException(e,sys,logger)
        
    def _create_validaton_report(self,flag:bool):
        '''creates the validation_report/validaton_report.yaml file containing the bool data is data validated or not
            
            args:
                flag(bool): all data valid or not according to schema
            
            
        '''

        dir = 'validation_report'
        os.makedirs(dir,exist_ok=True)

        validation_report_path = os.path.join(dir,'validation_report.yaml')

        validation_report = {
            'is_data_validated': flag
        }

        with open(validation_report_path,'w') as file:
            yaml.dump(validation_report,file)
           

    def validate_all_datasets_schema(self):
        ''' 
        Validates schema of datasets
        For Each Dataset:
            - load csv from artifact dir
            - loads corresponding yaml schema file
            - perform validation methods
            
        '''
        
        try:
            self._verify_artifact_data()
            df_schema_mappings = self.data_validation_config.load_dataset_schema_mapping()
            
            flag = True
            for dataset_schema in df_schema_mappings:
                self.status = True
                self.error_msg = ''
                
                df_file_path = os.path.join(self.data_ingestion_config.artifact_raw_dir,dataset_schema['data_path'])

                df = pd.read_csv(df_file_path)
                df.name = dataset_schema['name']

                schema_file = read_yaml_file( dataset_schema['schema_path'],logger)
                

                self.check_dataframe_schema(df,schema_file)
                self.validate_columns_dtypes(df,schema_file)


                if self.status == True:
                    logger.info(f' {df.name} :  Dataframe schema and Columns DataTypes Validation Success')
                    
                else:
                    logger.error(f'{df.name } : Errors Occuered While Validation Are: {self.error_msg}')
                    flag = False
            #this is used in the data_transformation module to ensure that the data is validated and the pipeline will not brok in
            #data transformation
            self._create_validaton_report(flag)

            if flag:
                logger.info(f'All datasets match expected schema')
            else:
                logger.error(f'All datasets doesnt match expected schema')

            self._create_validaton_report(flag)

                
        except Exception as e:
            raise MyException(e,sys,logger)
        
if __name__== '__main__':
   dv = DataValidation(DataValidationConfig(logger),DataIngestionConfig)
   dv.validate_all_datasets_schema()

 


