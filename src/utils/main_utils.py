import pandas as pd
import pandera as pa
import yaml 
from src.exception import MyException
import os
import sys

def read_yaml_file(yaml_file_path:str,logger:object):
    '''read and loads the yaml file from given path
      
    args:
        yaml_file_path: path of yaml file which we want to load
        logger: logger object to keep logs
    return:
        yaml_file
    '''
    try:
        with open(yaml_file_path,'r') as f:
            file = yaml.safe_load(f)
        logger.info(f'Yaml file load from {yaml_file_path} succesfully')
        return file
    except FileNotFoundError as e:
        raise MyException(e,sys,logger)
    
def save_object(obj:object,output_path:str):
    #write this function when will be required
    '''it will save the object in output_path can be model/object'''
    pass


def genrate_schema_dataset(name:str,df: pd.DataFrame,output_path:str):
    '''this function creates the schema yaml file for the df and stores in the config dir
        args:
            name: name for yaml file which will be created
            df: dataframe of which u want to create a schema
            output_path: where to save this yaml schema file
        return:
            save the schema for that dataset in the output_path
    '''
    #create a new name+yaml file
    file = os.path.join(output_path,name+'.yaml')

    with open(file,'w') as f:
        #name+_columns:
        f.write('columns:')
        schema = pa.infer_schema(df)
        for feature_name,data_type in schema.dtypes.items():
            f.write(f'\n  - {feature_name}: {data_type}')

        f.write('\n')
        f.write('\n')
        #name+_categorical_columns
        f.write('categorical_columns:')
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        for feature_name in categorical_columns:
            f.write(f'\n  - {feature_name}')
        
        f.write('\n')
        f.write('\n')
        #name+_numerical_columns
        f.write('numerical_columns:')

        numerical_columns = df.select_dtypes(exclude=['object', 'category', 'bool']).columns.tolist()
        for feature_name in numerical_columns:
            f.write(f'\n  - {feature_name}')

        
