import os

""" DATA INGESTION Releated Constant"""

RAW_DATA_URL = r"D:\credit risk datasets\datas" 


ARTIFACT_DIR = r'artifact'
ARTIFACT_RAW_DIR = os.path.join(ARTIFACT_DIR,'raw')
os.makedirs(ARTIFACT_RAW_DIR,exist_ok=True)    

# DATASET NAMES OF ALL from RAW_DATA_URL
APPLICATION_DATA = "application_data.csv"
BUREAU_DATA = "bureau.csv"
BUREAU_BALANCE_DATA = "bureau_balance.csv"
CREDIT_CARD_BALANCE = "credit_card_balance.csv"
INSTALLMENT_PAYMENTS_DATA = "installments_payments.csv"
POS_CASH_DATA = "POS_CASH_balance.csv"
PREVIOUS_APPLICATION_DATA  = "previous_application.csv"


