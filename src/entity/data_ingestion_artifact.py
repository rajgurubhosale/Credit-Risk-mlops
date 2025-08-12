import os
from src.constants import *
from src.constants.data_ingestion_constants import *
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:

    #artifact_raw_dir path 
    artifact_raw_dir:str = ARTIFACT_RAW_DIR
    source_raw_path: str = RAW_DATA_URL

    #path of all datas
    application_data:str = APPLICATION_DATA
    bureau_data: str = BUREAU_DATA
    bureau_balance_data: str = BUREAU_BALANCE_DATA
    credit_card_balance: str = CREDIT_CARD_BALANCE
    installment_payments_data: str = INSTALLMENT_PAYMENTS_DATA
    pos_cash_data: str = POS_CASH_DATA
    previous_application_data: str = PREVIOUS_APPLICATION_DATA

    def __post_init__(self):
        os.makedirs(ARTIFACT_RAW_DIR,exist_ok=True)

    def load_all_dataset(self):
        ''' 
        loads all the datasets file path
        
        return:
            file_list: list containg the file of the datasets to load
        '''
        file_list = [
                self.application_data,
                self.bureau_balance_data,
                self.bureau_data,
                self.credit_card_balance,
                self.installment_payments_data,
                self.pos_cash_data,
                self.previous_application_data
            ]
        return file_list
    



    




