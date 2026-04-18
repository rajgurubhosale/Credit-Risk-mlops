import pandas as pd
import pandera as pa
import yaml 
from src.exception import MyException
from pathlib import Path
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
    


def downcast_df_variables(df: Path) -> pd.DataFrame:
    """Load data and optimize memory usage by downcasting numeric types."""
    
    try:
        # Load data

        id_cols = [
            "SK_ID_CURR",
            "SK_ID_BUREAU",
            "SK_ID_PREV"
        ]


        int_cols = df.select_dtypes(include=["int64"]).columns

        # Exclude ID columns
        int_cols = [col for col in int_cols if col not in id_cols]

        df[int_cols] = df[int_cols].apply(
            pd.to_numeric,
            downcast="integer"
        )


        float_cols = df.select_dtypes(include=["float64"]).columns

        df[float_cols] = df[float_cols].apply(
            pd.to_numeric,
            downcast="float"
        )
        obj_cols = df.select_dtypes(include=['object']).columns
        df[obj_cols] = df[obj_cols].astype('category')

        return df

    except Exception as e:
        raise Exception(f"Error loading data from {e}")
