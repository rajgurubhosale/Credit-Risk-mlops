import os
from src.constants import *
from src.constants.data_ingestion_constants import *
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    #path of all datas
    application_data: str = os.path.join(RAW_DATA_URL,APPLICATION_DATA)
    bureau_data: str = os.path.join(RAW_DATA_URL,BUREAU_DATA)
    bureau_balance_data: str = os.path.join(RAW_DATA_URL,BUREAU_BALANCE_DATA)
    credit_card_balance: str = os.path.join(RAW_DATA_URL,CREDIT_CARD_BALANCE)
    installment_payments_data: str = os.path.join(RAW_DATA_URL,INSTALLMENT_PAYMENTS_DATA)
    pos_cash_data: str = os.path.join(RAW_DATA_URL,POS_CASH_DATA)
    previous_application_data: str = os.path.join(RAW_DATA_URL,PREVIOUS_APPLICATION_DATA)


    #save dataset of paths  
    save_application_data: str = os.path.join(ARTIFACT_RAW_DIR,APPLICATION_DATA)
    save_bureau_data: str = os.path.join(ARTIFACT_RAW_DIR,BUREAU_DATA)
    save_bureau_balance_data: str = os.path.join(ARTIFACT_RAW_DIR,BUREAU_BALANCE_DATA)
    save_credit_card_balance: str = os.path.join(ARTIFACT_RAW_DIR,CREDIT_CARD_BALANCE)
    save_installment_payments_data: str = os.path.join(ARTIFACT_RAW_DIR,INSTALLMENT_PAYMENTS_DATA)
    save_pos_cash_data: str = os.path.join(ARTIFACT_RAW_DIR,POS_CASH_DATA)
    save_previous_application_data: str = os.path.join(ARTIFACT_RAW_DIR,PREVIOUS_APPLICATION_DATA)



