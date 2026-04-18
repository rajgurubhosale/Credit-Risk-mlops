import pandas as pd
import sys
import yaml
from pathlib import Path
from src.logger import config_logger
from  src.utils.main_utils import *
from src.entity.artifact_entity import DataValidationArtifact
from src.constants import *
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants.artifacts_paths import *
from src.exception import MyException
import gc
logger = config_logger('module_02_data_validation.py')

class DataValidation:
    """
    Validates critical business rules and pipeline dependencies.
    Checks only what downstream feature engineering actually depends on.
    Raises on failure — pipeline halts, nothing is silently skipped.
    """

    def __init__(
        self,
        validation_artifact: DataValidationArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
        ):
        self.validation_artifact = validation_artifact
        self.data_ingestion_artifact = data_ingestion_artifact
        self.validation_report = {}
        
    
    def _report_validation_true(self,database_name):        
        ''' helper function just report validation status =True with data base name'''
        self.validation_report[database_name] = True
    
    def _report_validation_false(self,e,database_name):
        ''' helper function just report validation status = False with data base name'''
        
        self.validation_report[database_name] = {
                "status": False,
                "error": str(e)
            }
    
        
    def check_application(self):
        '''data validation check for the application data'''
        database_name = 'application_data'

        try:
            application_df = pd.read_csv(self.data_ingestion_artifact.application_data)
            
            assert application_df.shape[0]> 0, "Application Dataset is empty"
            # critical columns check
            assert "TARGET" in application_df.columns,"TARGET column missing"
            assert "SK_ID_CURR" in application_df.columns,"SK_ID_CURR missing"
            
            assert application_df["TARGET"].notna().all(),"TARGET contains null values"

            assert application_df["SK_ID_CURR"].notna().all(),"SK_ID_CURR contains null values"

            assert application_df["SK_ID_CURR"].is_unique,"Duplicate SK_ID_CURR found"
        
            self._report_validation_true(database_name)

            logger.info("Application dataset validation passed")
            
          
        except Exception as e:
            self._report_validation_false(e,database_name)
            raise MyException(e,sys,logger)
        finally:
            del application_df
            gc.collect()
            
    def check_bureau(self):
        '''data validation check for the bureau data'''
        database_name ='bureau'

        try:
            bureau_df = pd.read_csv(self.data_ingestion_artifact.bureau_data)
            
            assert bureau_df.shape[0] >0 ,"Bureau dataset is empty"
            assert "SK_ID_BUREAU" in bureau_df.columns, "SK_ID_BUREAU missing"
            assert "SK_ID_CURR" in bureau_df.columns,"SK_ID_CURR missing in bureau"
            assert bureau_df["SK_ID_CURR"].notna().all(),"Null SK_ID_CURR in bureau"
            self._report_validation_true(database_name)
            logger.info("Bureau dataset validation passed")
       
        except Exception as e:
            self._report_validation_false(e,database_name)
            
            raise MyException(e,sys,logger)
        finally:
            del bureau_df
            gc.collect()
            
    
    def check_bureau_balance(self):

        database_name = 'bureau_balance'
        
        try:
            df = pd.read_csv(self.data_ingestion_artifact.bureau_balance)

            REQUIRED_COLS = [
                "SK_ID_BUREAU",
                "MONTHS_BALANCE",
                "STATUS"
            ]

            for col in REQUIRED_COLS:
                assert col in df.columns,f"{col} missing — DPD features will fail"

            assert df.shape[0] > 0,"bureau_balance dataset empty"

            self._report_validation_true(database_name)

            logger.info("Bureau Balance validation passed")

        except Exception as e:
            self._report_validation_false(e,database_name)

            raise MyException(e,sys,logger)
        finally:
            del df
            gc.collect()
    def check_installment_payments(self):
        '''data validation check for the installment payments data'''
        database_name = 'installments_payments'

        try:
            df = pd.read_csv(
            self.data_ingestion_artifact.installment_payments_data,
            on_bad_lines="skip",

            )
            
            assert "SK_ID_CURR" in df.columns,"SK_ID_CURR missing"
            assert "SK_ID_PREV" in df.columns,"SK_ID_PREV missing"
            assert "NUM_INSTALMENT_VERSION" in df.columns,"NUM_INSTALMENT_VERSION missing"
            assert df.shape[0]> 0,"dataframe is empty"
            self._report_validation_true(database_name)

            logger.info("installments payments validation passed")

            
        except Exception as e:
            self._report_validation_false(e,database_name)
            raise MyException(e,sys,logger)
        finally:
            del df
            gc.collect()
    def check_previous_applications(self):
        '''data validation check fo r the previos applications data'''
        database_name = 'previous_application'

        try: 
            df = pd.read_csv(self.data_ingestion_artifact.previous_application_data)
            
            assert "SK_ID_CURR" in df.columns,"SK_ID_CURR missing"
            assert "SK_ID_PREV" in df.columns,"SK_ID_PREV missing" 
            assert df.shape[0]> 0,"dataframe is empty"
            self._report_validation_true(database_name)


            
            logger.info("previous_application  dataframe validation passed")

      
        except Exception as e:
            self._report_validation_false(e,database_name)

            raise MyException(e,sys,logger)
        finally:
            del df
            gc.collect()
    def check_pos_cash_balance(self):
        '''data validation check for the previos pos cash balance data'''
        database_name = 'POS_CASH_balance'
        
        try:
            df = pd.read_csv(self.data_ingestion_artifact.pos_cash_data)
            assert "SK_ID_CURR" in df.columns,"SK_ID_CURR missing"
            assert "SK_ID_PREV" in df.columns,"SK_ID_PREV missing" 
            assert df.shape[0]> 0,"dataframe is empty"
            assert "MONTHS_BALANCE" in df.columns,"MONTHS_BALANCE missing"
            self._report_validation_true(database_name)


            logger.info("Pos cash balance dataframe validation passed")
          
        except Exception as e:
            self._report_validation_false(e,database_name)

            raise MyException(e,sys,logger)
        finally:
            del df
            gc.collect()
    def check_credit_balance(self):
        '''data validation check for the previos credit balance data'''
        database_name = 'credit_card_balance'

        try:
            df = pd.read_csv(self.data_ingestion_artifact.credit_card_balance)

            assert "SK_ID_PREV" in df.columns,"SK_ID_PREV missing"
            assert "SK_ID_CURR" in df.columns,"SK_ID_CURR missing"
            assert "AMT_BALANCE" in df.columns,"AMT_BALANCE missing"
            assert df.shape[0]> 0,"dataframe is empty"
            self._report_validation_true(database_name)

            logger.info("Credit balance validation passed")
            
        except Exception  as e:
            self._report_validation_false(e,database_name)
            raise MyException(e,sys,logger)
        finally:
            del df
            gc.collect()
    def save_report(self):

        report_path = self.validation_artifact.validation_report_path

        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate final status
        all_valid = all(
            value == True
            for value in self.validation_report.values()
        )

        self.validation_report['is_data_validated'] = all_valid

        with open(report_path, "w") as file:
            yaml.dump(self.validation_report, file)

        logger.info("Validation report saved")
        
    def initiate_data_validation(self):

        methods = [ 
            self.check_application,
            self.check_bureau,
            self.check_bureau_balance,
            self.check_installment_payments,
            self.check_pos_cash_balance,
            self.check_previous_applications,
            self.check_credit_balance
        ]

        try:
            for method in methods:
                method()

        except Exception as e:
            logger.error("Validation failed")
            self.validation_report['is_data_validated'] = False

        finally:
            # ALWAYS create output file
            self.save_report()
            
if __name__ == '__main__':
    dv = DataValidation(DataValidationArtifact(),DataIngestionArtifact())
    dv.initiate_data_validation()