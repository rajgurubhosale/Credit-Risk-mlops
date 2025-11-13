import os
import sys
import numpy as np
import pandas as pd
from abc import ABC,abstractmethod

from src.entity.data_ingestion_artifact import DataIngestionConfig
from src.entity.data_transformation_artifact import DataTransformationConfig
from src.utils.main_utils import read_yaml_file
from src.logger import *
from src.exception import *        

from src.entity.data_ingestion_artifact import * 
from src.entity.data_transformation_artifact import *
from src.entity.data_ingestion_artifact import *
from src.entity.data_validation_artifact import *
from src.constants.data_transformation_constant import *


logger = config_logger('03_data_transformation')

class ApplicationDfTransformer:
    '''basic preprocessing of application main dataset
        - preprocess df
        - create days to year features
        - handle place holder values such as xNA/XAP 
     '''
    def __init__(self,main_df_path:str,config: ApplicationDfConfig):
        '''load the dataset in self.main_df
        
            args:
                main_df_path: application dataset path
        
        '''
        self.config = config
        self.main_df = pd.read_csv(main_df_path,dtype=self.config.application_data_dtypes_reduce)


    def _preprocessing(self):
        '''
        simplifying values in Application Df
        '''
        try:
            for col, mapping in self.config.simplify_values_mapping.items():
                if col in self.main_df.columns:
                    self.main_df[col] = self.main_df[col].replace(mapping)
                else:
                    logger.debug(f'{col} : column is not present in the DataFrame')
        except Exception as e:
            raise MyException(e,sys,logger)


    def _handle_invalid_values(self):
        '''
        Convert placeholder values to nan

        placeholder_values :
            - XNA / XAP / UNknown / 365243

        '''
        try:
            # feature wise handling local placeholders
            for col , mapping in self.config.placeholders_mapping['local_placeholder'].items():
                if col in self.main_df.columns:
                    self.main_df[col] = self.main_df[col].replace(mapping)
                else:
                    logger.debug(f'{col} : column is not present in the DataFrame')

            # global placeholders
            self.main_df = self.main_df.replace(self.config.placeholders_mapping['global_placeholders'])
            
        except Exception as e:

            raise MyException(e,sys,logger)

    def _convert_days_to_years(self):
        '''Converts Features values Days to Years

            Feature Transformed:
            - DAYS_BIRTH : YEARS_AGE
            - DAYS_EMPLOYED : YEARS_EMPLOYED
            - DAYS_REGISTRATION : YEARS_REGISTRATION
            - DAYS_ID_PUBLISH :  YEARS_ID_PUBLISH

            Converting -0.0  to 0.0 value for  consistency 
            Drop the days original columns

        '''
        try:
            for col,trasform_col in self.config.days_to_years_mapping.items():

                if col in self.main_df.columns:
                    #converting the days feature to the years
                    self.main_df[trasform_col] = round( -self.main_df[col] / 365, 2)    

                    #converting new feature to float32 to reduce space 
                    self.main_df[trasform_col] = self.main_df[trasform_col].astype('float32')

                    #converting -0.0  to 0.0 value for the consistency 
                    self.main_df[trasform_col] = self.main_df[trasform_col].replace(-0.0,0.0)
                    
                    #dropped the days columns that is converted.it contains same information.
                    self.main_df.drop(columns = [col],inplace=True)
                else:
                    logger.debug(f'{col} : column is not present in the DataFrame')

        except Exception as e:
            raise MyException(e,sys,logger)

    def run_preprocessing_steps(self):

        '''run all preprocessing steps in sequence

            return:
                self.main_df: returns the dataframe after all preprocessing
            '''
        try:
            self._preprocessing()
            self._handle_invalid_values()
            self._convert_days_to_years()

            logger.info('Application DataFrame preprocessing done successfully')

        except Exception as e:
            raise MyException(e,sys,logger)

        return self.main_df
    



class BaseTransformer(ABC):
    '''combine all the diff dataset into one single dataframe for analysis and model pipeline'''

    def __init__(self,data_transformation_config:DataTransformationConfig,data_ingestion_config:DataIngestionConfig):

        self.data_transformation_config = data_transformation_config
        self.data_ingestion_config = data_ingestion_config

    def load_data(self,data_path:str,dtypes:dict=None,): #pass the data.csv like name from the scema.yaml
        ''' load data in reduced dtypes format to optimize memory'''
        try:
            # load data from the artifact/raw
            df_path = os.path.join(self.data_ingestion_config.artifact_raw_dir,data_path)

            return pd.read_csv(df_path,dtype = dtypes)
        except Exception as e:
            raise MyException(e,sys,logger)
        

    @abstractmethod
    def add_features_main(self,main_df=None):
        '''create features and append in main dataframe'''
        pass
    

  

class BureauBalanceTransformation(BaseTransformer):
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_config: DataIngestionConfig):
       super().__init__(data_transformation_config, data_ingestion_config)

       self.bureau_balance = self.load_data(
           self.data_ingestion_config.bureau_balance_data,
           self.data_transformation_config.bureau_balance_dtypes_reduce
       )
       self.bureau = self.load_data(
           self.data_ingestion_config.bureau_data,
           self.data_transformation_config.bureau_dtypes_reduce
       )
       
    def _extract_status(self):
        ''' Extract features from the 'STATUS column in the bureau dataframe.
    
            Features Transformed:
            - MAX_DPD: max dpd of the client from all the credit loans

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
       # encoding values in numbers
        if 'STATUS' in self.bureau_balance.columns:
            dpd_map = {'X':np.nan,'C':0,'1':1,'2':2,'3':3,'4':4,'5':5}
            self.bureau_balance['STATUS'] = self.bureau_balance['STATUS'].map(dpd_map)

            max_dpd = self.bureau_balance.groupby(by='SK_ID_BUREAU')['STATUS'].max().to_frame()
            
            max_dpd = max_dpd.rename(columns={'STATUS':'MAX_DPD'})

            self.bureau = self.bureau.merge(max_dpd,on='SK_ID_BUREAU',how='left')

            features_df = self.bureau.groupby('SK_ID_CURR')['MAX_DPD'].max().to_frame()

            return features_df
        else:
            logger.debug('STATUS : column is not present in the DataFrame')


    def add_features_main(self, main_df):
        '''Extract and Create feature from bureau balance dataset and append in main dataframe'''

        try:
            self.feature_extractors  = [
                self._extract_status
            ]

            self.main_df  = main_df

            for extractor in self.feature_extractors:
                features_df = extractor()
                self.main_df = self.main_df.merge(features_df,on='SK_ID_CURR',how='left')
            logger.info("Aggregated features from the BUREAU BALANCE dataframe successfully merged into the main dataframe.")

            return self.main_df
        
        except Exception as e:
            raise MyException(e,sys,logger)
 



class BureauTransformer(BaseTransformer):
    '''Extracts and Transform features from the bureau data. And append in the main dataframe'''
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_config: DataIngestionConfig):
        '''Load the bureau dataset and save in the self.df dataframe'''
        super().__init__(data_transformation_config, data_ingestion_config)

        self.df = self.load_data(
            data_path = self.data_ingestion_config.bureau_data,
            dtypes = self.data_transformation_config.bureau_dtypes_reduce
        )

    def _extract_credit_active(self):

        ''' Extract features from the 'CREDIT_ACTIVE' column in the bureau dataframe.
            
            Features Transformed:
            - HAS_BAD_LOAN : 1 if the customer has any bad or sold loan, else 0
            - NUM_ACTIVE_CREDIT : Count of active credits for the customer.
            - NUM_CLOSED_CREDIT: Count of closed credits for the customer.

            Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''
        if 'CREDIT_ACTIVE' in self.df.columns:

            crosstab = pd.crosstab(self.df['SK_ID_CURR'], self.df['CREDIT_ACTIVE'],dropna=False)

            if any(pd.isna(crosstab.columns)):
                idx = crosstab[crosstab[np.nan]==1].index
                crosstab.loc[idx] = np.nan

            #filtering and creating flags if person have bad loan
            crosstab['HAS_BAD_LOAN'] = ((crosstab['Bad debt'] > 0) | (crosstab['Sold']> 0)).astype(int)

            features_df = crosstab[['Active','HAS_BAD_LOAN','Closed']].copy()

            features_df = features_df.rename(columns={'Active':'NUM_ACTIVE_CREDIT','Closed':'NUM_CLOSED_CREDIT'})

            return features_df

        else:
            logger.debug('CREDIT_ACTIVE : column is not present in the DataFrame')

        
    def _extract_days_credit(self):

        ''' Extract features from the 'DAYS_CREDIT' column in the bureau dataframe.
            
            Features Transformed:
            - RECENT_CREDIT_FLAG_90D: Flag if the customer had taken credit in last 90 days.
            - NUM_ACTIVE_CREDIT_180D: Count of the active credits in last 180 Days.

            Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''

        if 'DAYS_CREDIT' in self.df.columns:

            # 1 : RECENT_CREDIT_FLAG_90D

            #creating new column and assign 0 values 
            self.df['RECENT_CREDIT_FLAG_90D'] = 0

            #assign flag to recent credit 90 days
            filt = self.df['DAYS_CREDIT'] >= -90
            self.df.loc[filt,'RECENT_CREDIT_FLAG_90D'] = 1

            # fill the nan values in place where the day credit is missing in that index for flag instead of 0
            filt = self.df['DAYS_CREDIT'].isnull()
            self.df.loc[filt,'RECENT_CREDIT_FLAG_90D'] = np.nan

            features_df_1 = self.df.groupby(by='SK_ID_CURR')['RECENT_CREDIT_FLAG_90D'].max().to_frame()

            # --------------------------------------------------------------
            
            # 2 : NUM_ACTIVE_CREDIT_180D

            #filtring the data of the latest 180 days credit
            filt = self.df['DAYS_CREDIT'] >= -180
            temp = self.df.loc[filt].copy()
            
            crosstab = pd.crosstab(temp['SK_ID_CURR'],temp['CREDIT_ACTIVE'],dropna=False)

            # keeping the null            
            if any(pd.isna(crosstab.columns)):
                idx = crosstab[crosstab[np.nan]==1].index
                crosstab.loc[idx] = np.nan

            features_df_2 = crosstab['Active'].to_frame().rename(columns={'Active':'NUM_ACTIVE_CREDIT_180D'})
            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            features_df_2 = features_df_2.reindex(all_cust_index, fill_value=0)
            features_df = features_df_1.merge(features_df_2,on='SK_ID_CURR',how='outer')

            return features_df
        
        else:
            logger.debug('DAYS_CREDIT : column is not present in the DataFrame')

       
    def _extract_credit_day_overdue(self):

        ''' Extract features from the 'CREDIT_DAY_OVERDUE' column in the bureau dataframe.
            
            Features Transformed:
            - HAS_CREDIT_DAYS_OVERDUE: Flag if customer has any overdue credit days 1,0

            Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''

        if 'CREDIT_DAY_OVERDUE' in self.df.columns:
                
            # create a flag of overdue
            self.df['HAS_CREDIT_DAYS_OVERDUE'] = 0
            filt = self.df['CREDIT_DAY_OVERDUE'] != 0
            self.df.loc[filt,'HAS_CREDIT_DAYS_OVERDUE'] = 1

            # assgin nan if credit_day_overdue feature had nan values to flag feature
            filt = self.df['CREDIT_DAY_OVERDUE'].isnull()
            self.df.loc[filt,'HAS_CREDIT_DAYS_OVERDUE'] = np.nan

            # aggreagate
            features_df = self.df.groupby('SK_ID_CURR')['HAS_CREDIT_DAYS_OVERDUE'].max().to_frame()

            return features_df
        else:
            logger.debug('CREDIT_DAY_OVERDUE : column is not present in the DataFrame')


    def _extract_days_enddate(self):

        ''' Extract features from the 'DAYS_ENDDATE_FACT' and 'DAYS_CREDIT_ENDDATE' column in the bureau dataframe.
            
            Features Transformed:
            - AVG_REPAYMENT_DAYS_DIFF :Average diff in days between actual and scheduled credit end date for closed credits.

            Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''
                
        if ('DAYS_ENDDATE_FACT' in self.df.columns) and ('DAYS_CREDIT_ENDDATE' in self.df.columns):

            filt = self.df['CREDIT_ACTIVE'] == 'Closed'
            self.df.loc[filt,'REPAYMENT_DAYS_DIFF'] = self.df['DAYS_ENDDATE_FACT'] - self.df['DAYS_CREDIT_ENDDATE']
            
            features_df = self.df.groupby(by='SK_ID_CURR')['REPAYMENT_DAYS_DIFF'].mean().to_frame()
            features_df = features_df.rename(columns={'REPAYMENT_DAYS_DIFF':'AVG_REPAYMENT_DAYS_DIFF'})
            
            return features_df
        else:
            logger.debug('DAYS_ENDDATE_FACT, DAYS_CREDIT_ENDDATE: column is not present in the dataframe')


    def  _amt_credit_max_overdue(self):

        ''' Extract features from the 'AMT_CREDIT_MAX_OVERDUE' column in the bureau dataframe.
        
            Features Transformed:
            - FLAG_HAS_AMT_OVERDUE: flag if the customer had any amt overdue 1, 0
            Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''

        if 'AMT_CREDIT_MAX_OVERDUE' in self.df.columns:        
            self.df['FLAG_HAS_AMT_OVERDUE'] = np.where(
                self.df['AMT_CREDIT_MAX_OVERDUE'] > 0,
                1,
                0
            )
            filt = self.df['AMT_CREDIT_MAX_OVERDUE'].isnull()
            self.df.loc[filt,'FLAG_HAS_AMT_OVERDUE'] = np.nan
            
            features_df = self.df.groupby(by='SK_ID_CURR')['FLAG_HAS_AMT_OVERDUE'].max().to_frame()
            
            return features_df
        else:
            logger.debug('AMT_CREDIT_MAX_OVERDUE: column is not present in the DataFrame')


    def _cnt_credit_prolong(self):
        ''' Extract features from the 'CNT_CREDIT_PROLONG' column in the bureau dataframe.
    
            Features Transformed:
            - FLAG_HAS_CREDIT_PROLONG: flag if the customer credit prolong

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'CNT_CREDIT_PROLONG' in self.df.columns:

            self.df['FLAG_HAS_CREDIT_PROLONG'] = np.where(    
            self.df['CNT_CREDIT_PROLONG'].isnull(),
            np.nan,
            (self.df['CNT_CREDIT_PROLONG'] > 0).astype(int)
            )
            
            features_df =self.df.groupby(by='SK_ID_CURR')['FLAG_HAS_CREDIT_PROLONG'].max().to_frame()
            return features_df
           
        else:
            logger.debug('CNT_CREDIT_PROLONG: column is not present in the DataFrame')

    def _extract_features_amt_credit(self):
        ''' Extract features from the 'AMT_CREDIT_SUM_DEBT' and 'AMT_CREDIT_SUM' column in the bureau dataframe.
    
            Features Transformed:
            - DEBT_TO_LOAN_RATIO: ratio of total debt to total credit of customer
            - AVG_AMT_CREDIT_SUM: averge of all credit amount of the customer

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''


        if ('AMT_CREDIT_SUM_DEBT' in self.df.columns) and ('AMT_CREDIT_SUM' in self.df.columns):

            #loan to debt ratio
            self.df['DEBT_TO_LOAN_RATIO'] = np.where(
                (self.df['CREDIT_ACTIVE'] =='Active') & (self.df['AMT_CREDIT_SUM'] !=0),
                self.df['AMT_CREDIT_SUM_DEBT'] / self.df['AMT_CREDIT_SUM'],
                0
                )
            filt = (self.df['AMT_CREDIT_SUM'].isnull()) | (self.df['AMT_CREDIT_SUM_DEBT'].isnull())

            self.df.loc[filt,'DEBT_TO_LOAN_RATIO'] = np.nan

            features_df = self.df.groupby(by='SK_ID_CURR')['DEBT_TO_LOAN_RATIO'].mean().to_frame()

            # avg loan amount of previous all options!
            features_df_2 = self.df.groupby(by='SK_ID_CURR')['AMT_CREDIT_SUM'].mean().to_frame()
            features_df_2 = features_df_2.rename(columns={'AMT_CREDIT_SUM':'AVG_AMT_CREDIT_SUM'})

            features_df = features_df.merge(features_df_2,on='SK_ID_CURR',how='outer')

            return features_df

        else:
            logger.debug('DEBT_TO_LOAN_RATIO ,AMT_CREDIT_SUM : column is not present in the DataFrame')

    def _extract_has_credit_loan(self):
        ''' Extract features from the 'CREDIT_TYPE column in the bureau dataframe.
    
            Features Transformed:
            - HAS_CREDIT_LOAN: person have or had the credit loan

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        
        if 'CREDIT_TYPE' in self.df.columns:
            self.df['HAS_CREDIT_LOAN'] = np.where(
            self.df['CREDIT_TYPE'] == 'Credit card',
                1,
                0
            )
            filt = self.df['CREDIT_TYPE'].isnull()
            self.df.loc[filt,'HAS_CREDIT_LOAN'] = np.nan
            
            features_df = self.df.groupby('SK_ID_CURR')['HAS_CREDIT_LOAN'].max().to_frame()

            return features_df
        else:
            logger.debug('CREDIT_TYPE : column is not present in the DataFrame')

    def add_features_main(self,main_df):

        '''Extract and Create feature from bureau dataset and append in main dataframe'''
        try:
            # private functions to run so it will extract feature and create a df in loop
            self.feature_extractors   = [
                self._extract_credit_active,
                self._extract_days_credit,
                self._extract_credit_day_overdue,
                self._extract_days_enddate,
                self._extract_features_amt_credit,
                self._extract_has_credit_loan
                ]

            self.main_df  = main_df

            for extractor in self.feature_extractors:
                features_df = extractor()
                self.main_df = self.main_df.merge(features_df,on='SK_ID_CURR',how='left')

            logger.info("Aggregated features from the BUREAU dataframe successfully merged into the main dataframe.")

            return self.main_df

        except Exception as e:
            raise MyException(e,sys,logger)



class InstallmentsPaymentsTransformation(BaseTransformer):
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_config: DataIngestionConfig):
        super().__init__(data_transformation_config, data_ingestion_config)


        self.df = self.load_data(
            self.data_ingestion_config.installment_payments_data,
            self.data_transformation_config.installment_payment_dtypes_reduce
            )
      


    def _extract_number_of_reshedules_6m(self):
        ''' Extract features from the 'NUM_INSTALMENT_VERSION' column in the installments payments dataframe.
    
            Features Transformed:
            - NUMBER_OF_RESHEDULES_6M: the number of times the loan is resheduled in the recent 6 month of period

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NUM_INSTALMENT_VERSION' in self.df.columns and 'DAYS_INSTALMENT' in self.df.columns:
            
            filt = (self.df['DAYS_INSTALMENT'] >= -180) & (self.df['NUM_INSTALMENT_VERSION']!=0)

            filt_df_6m = self.df.loc[filt]
            temp = (filt_df_6m.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].max()-1).to_frame()
            
            temp = temp.groupby('SK_ID_CURR').sum()
            
            temp = temp.rename(columns={'NUM_INSTALMENT_VERSION':'NUMBER_OF_RESHEDULES_6M'})
            
            # assign 0 where the num_installment dats are greater than 180 or number of reshedule is 0
            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = temp.reindex(all_cust_index,fill_value=0)

            return feature_df
        
        else:
            logger.debug('NUM_INSTALMENT_VERSION & DAYS_INSTALMENT : column is not present in the DataFrame')


    def _extract_mean_dpd_2Y(self):
        ''' Extract features from the 'DAYS_ENTRY_PAYMENT and DAYS_ENTRY_PAYMENT column in the installments payments dataframe.
    
            Features Transformed:
            - MEAN_DPD_RECENT_2Y: Average DPD of the customer that considering that loans is payed late in recent 2 Years

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'DAYS_ENTRY_PAYMENT' in self.df.columns and 'DAYS_INSTALMENT' in self.df.columns:
            
            self.df['DPD'] = self.df['DAYS_ENTRY_PAYMENT']- self.df['DAYS_INSTALMENT']
            filt  = (self.df['DPD']> 0) & (self.df['DAYS_INSTALMENT'] > -730)
            filt_df_2y = self.df.loc[filt]

            temp = filt_df_2y.groupby(['SK_ID_CURR','SK_ID_PREV'])['DPD'].sum().to_frame()
            feature_df = temp.groupby('SK_ID_CURR')['DPD'].mean().to_frame()
            feature_df = feature_df.rename(columns={'DPD':'MEAN_DPD_RECENT_2Y'})

            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index,fill_value=0)
            return feature_df
        
        else:
            logger.debug('DAYS_ENTRY_PAYMENT or DAYS_INSTALMENT : column is not present in the DataFrame')


    def _extract_sum_dpd_recent_6m(self):
        ''' Extract features from the 'DAYS_ENTRY_PAYMENT and DAYS_ENTRY_PAYMENT column in the installments payments dataframe.
    
            Features Transformed:
            - SUM_DPD_RECENT_6M: Sum dpd of the customer that considering that loans is payed late in recent 6 Months

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'DAYS_ENTRY_PAYMENT' in self.df.columns and 'DAYS_INSTALMENT' in self.df.columns:

            if 'DPD' not in self.df.columns:
                self.df['DPD'] = self.df['DAYS_ENTRY_PAYMENT']- self.df['DAYS_INSTALMENT']

            filt  = (self.df['DPD']> 0) & (self.df['DAYS_INSTALMENT'] > -180)   
            filt_df_6m = self.df.loc[filt]

            feature_df = filt_df_6m.groupby('SK_ID_CURR')['DPD'].sum().to_frame()
            feature_df = feature_df.rename(columns={'DPD':'SUM_DPD_RECENT_6M'})


            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index,fill_value=0)
            return feature_df
        
        else:
            logger.debug('DAYS_ENTRY_PAYMENT or DAYS_INSTALMENT : column is not present in the DataFrame')



    def _extract_num_underpaid_installments_1y(self):
        ''' Extract features from the 'AMT_INSTALMENT and AMT_PAYMENT column in the installments payments dataframe.
    
            Features Transformed:
            - NUM_UNDERPAID_INSTALLMENTS_1Y: number of underpaid installments of the client in recent 1 year

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
            '''
        
        if 'AMT_INSTALMENT' in self.df.columns and 'AMT_PAYMENT' in self.df.columns:

            filt  = (self.df['DAYS_INSTALMENT'] > -365)
            filt_df_1y = self.df.loc[filt]
            
            filt_df_1y = filt_df_1y.groupby(['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER'])[['AMT_INSTALMENT','AMT_PAYMENT']].sum()

            filt_df_1y['NUM_UNDERPAID_INSTALLMENTS'] = (filt_df_1y['AMT_INSTALMENT'] > filt_df_1y['AMT_PAYMENT']).astype('int')

            feature_df = filt_df_1y.groupby('SK_ID_CURR')['NUM_UNDERPAID_INSTALLMENTS'].sum().to_frame()

            feature_df =  feature_df.rename(columns={'NUM_UNDERPAID_INSTALLMENTS':'NUM_UNDERPAID_INSTALLMENTS_1Y'})
            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()

            feature_df = feature_df.reindex(all_cust_index,fill_value=0)
            return feature_df
        
        else:
            logger.debug('AMT_INSTALMENT or AMT_PAYMENT : column is not present in the DataFrame')



    def add_features_main(self,main_df):
        '''Extract and Create feature from installments payment dataset and append in previous application  dataframe'''   
        try:
            self.feature_extractors  = [
                self._extract_number_of_reshedules_6m,
                self._extract_sum_dpd_recent_6m,
                self._extract_mean_dpd_2Y,
                self._extract_num_underpaid_installments_1y
            ]

            for extractor in self.feature_extractors:
                features_df = extractor()
                main_df = main_df.merge(features_df,on='SK_ID_CURR',how='left')

            logger.info("Aggregated features from the installlments payments dataframe successfully merged into the Previous Application dataframe.")

            return main_df
         
        except Exception as e:
            raise MyException(e,sys,logger)
 



class PosCashBalanceTransformation(BaseTransformer):
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_config: DataIngestionConfig):
        super().__init__(data_transformation_config, data_ingestion_config)

        self.df = self.load_data(
            self.data_ingestion_config.pos_cash_data,
            self.data_transformation_config.pos_cash_reduce_dtypes
        )


    def _extract_has_risky_contract_status(self):
        ''' Extract features from the 'NAME_CONTRACT_STATUS  from the pos_cash_balance dataset.
    
            Features Transformed:
            - HAS_RISKY_CONTRACT_STATUS: flag the customer who had the NAME_CONTRACT_STATUS 'Demand','Amortized debt','Returned to the store'.

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
            '''
        
        if 'NAME_CONTRACT_STATUS' in self.df.columns :

            risky_categories = ['Demand','Amortized debt','Returned to the store']
            
            self.df['HAS_RISKY_CONTRACT_STATUS'] = np.where(self.df['NAME_CONTRACT_STATUS'].isin(risky_categories),
                    1,
                    0)

            feature_df = self.df.groupby('SK_ID_CURR')['HAS_RISKY_CONTRACT_STATUS'].max().to_frame()
            
            return feature_df
        
        else:
            logger.debug('NAME_CONTRACT_STATUS : column is not present in the DataFrame')


    def _extract_num_active_loans_6m(self):
        ''' Extract features from the 'NAME_CONTRACT_STATUS  from the pos_cash_balance dataset.
    
            Features Transformed:
            - NUM_ACTIVE_LOANS_6M: count of active loans in the last 6 months

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
            '''
        
        if 'NAME_CONTRACT_STATUS' in self.df.columns:

            filt_df_6m = self.df[(self.df['MONTHS_BALANCE'] >=-6) ]

            completed_loans_6m = filt_df_6m[filt_df_6m['NAME_CONTRACT_STATUS']=='Completed']['SK_ID_PREV'].unique()

            filt_df_6m = filt_df_6m[(filt_df_6m['NAME_CONTRACT_STATUS'] == 'Active') & ~(filt_df_6m['SK_ID_PREV'].isin(completed_loans_6m))]
            feature_df = filt_df_6m.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame()

            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index,fill_value=0)

            feature_df = feature_df.rename(columns={'SK_ID_PREV':'NUM_ACTIVE_LOANS_6M'})
        
            return feature_df
        
        else:
            logger.debug('NAME_CONTRACT_STATUS : column is not present in the DataFrame')
    
    def _extract_num_active_loans_1y(self):
        ''' Extract features from the 'NAME_CONTRACT_STATUS  from the pos_cash_balance dataset.
    
            Features Transformed:
            - NUM_ACTIVE_LOANS_1y: count of active loans in the last 1 year

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
            '''
        
        if 'NAME_CONTRACT_STATUS' in self.df.columns:

            filt_df_12m = self.df[(self.df['MONTHS_BALANCE'] >=-12) ]
            completed_loans_12m = filt_df_12m[filt_df_12m['NAME_CONTRACT_STATUS']=='Completed']['SK_ID_PREV'].unique()

            filt_df_12m = filt_df_12m[(filt_df_12m['NAME_CONTRACT_STATUS'] == 'Active') & ~(filt_df_12m['SK_ID_PREV'].isin(completed_loans_12m))]
            feature_df = filt_df_12m.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame()

            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index,fill_value=0)

            feature_df = feature_df.rename(columns={'SK_ID_PREV':'NUM_ACTIVE_LOANS_1Y'})
            
            return feature_df 

        else:
            logger.debug('NAME_CONTRACT_STATUS : column is not present in the DataFrame')

    def _extract_cnt_installment_future(self):
        ''' Extract features from the 'CNT_INSTALMENT_FUTURE  from the pos_cash_balance dataset.
    
            Features Transformed:
            - TOTAL_REMAINING_INSTALLMENTS: number of installement the customer had to pay. i.e reaminaing num installments 

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
            '''
        
        if 'CNT_INSTALMENT_FUTURE' in self.df.columns:

            filt_df = self.df[(self.df['MONTHS_BALANCE'] ==-1) & self.df['NAME_CONTRACT_STATUS'].isin(['Active','Signed'])]            
            feature_df = filt_df.groupby(by='SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].sum().to_frame()
            feature_df = feature_df.rename(columns={'CNT_INSTALMENT_FUTURE':'TOTAL_REMAINING_INSTALLMENTS'})
            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index,fill_value=0)
            return feature_df 

        else:
            logger.debug('CNT_INSTALMENT_FUTURE : column is not present in the DataFrame')


    def _extract_flag_dpd_def_1y(self):
    
        ''' Extract features from the 'SK_DPD_DEF  from the pos_cash_balance dataset.
    
            Features Transformed:
            - FLAG_DPD_DEF_1Y: flag is there is DPD_DEF in last 1 year

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
            '''
        
        if 'SK_DPD_DEF' in self.df.columns:

            self.df['FLAG_DPD_DEF_1Y'] = np.where((self.df['SK_DPD_DEF'] >  0) & (self.df['MONTHS_BALANCE'] >=- 12),
            1,
            0)

            feature_df = self.df.groupby('SK_ID_CURR')['FLAG_DPD_DEF_1Y'].max().to_frame()
            return feature_df 

        else:
            logger.debug('SK_DPD_DEF : column is not present in the DataFrame')



    def _extract_mean_dpd_1y(self):
    
        ''' Extract features from the 'SK_DPD_DEF  from the pos_cash_balance dataset.
    
            Features Transformed:
            - MEAN_DPD_1Y: mean of the dpd in last 1 year

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
            '''
        
        if 'SK_DPD' in self.df.columns:

            filt_df_1y = self.df[self.df['MONTHS_BALANCE'] >=- 12]
            feature_df = filt_df_1y.groupby('SK_ID_CURR')['SK_DPD'].mean().to_frame()
            feature_df = feature_df.rename(columns={'SK_DPD':'MEAN_DPD_1Y'})
            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index,fill_value=0)
            return feature_df 

        else:
            logger.debug('SK_DPD : column is not present in the DataFrame')
    
    def add_features_main(self,main_df):

        '''Extract and Create feature Pos Cash Balance and append in Main Dataframe'''   
        try:
            self.feature_extractors  = [
                self._extract_has_risky_contract_status,
                self._extract_num_active_loans_6m,
                self._extract_num_active_loans_1y,
                self._extract_cnt_installment_future,
                self._extract_flag_dpd_def_1y,
                self._extract_mean_dpd_1y
            ]

            for extractor in self.feature_extractors:
                features_df = extractor()
                main_df = main_df.merge(features_df,on='SK_ID_CURR',how='left')

            logger.info("Aggregated features from the pos cash balance dataframe successfully merged into the main  Application dataframe.")

            return main_df
         
        except Exception as e:
            raise MyException(e,sys,logger)




class PreviousApplicationsTransformation(BaseTransformer):

    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_config: DataIngestionConfig):
       super().__init__(data_transformation_config, data_ingestion_config)

       self.df = self.load_data(
           self.data_ingestion_config.previous_application_data,
           self.data_transformation_config.previous_application_dtypes_reduce
       )
       
    
    def _extract_avg_amt_annuity(self):
        ''' Extract features from the 'AMT_ANNUITY' in the previous application dataset.

            Features Transformed:
            - AVG_AMT_ANNUITY_CLIENT:  average amt annuity per client (avg regular installemtn amount)

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'AMT_ANNUITY' in self.df.columns :
            
            feature_df = self.df.groupby(by='SK_ID_CURR')['AMT_ANNUITY'].mean().to_frame()
            feature_df = feature_df.rename(columns={'AMT_ANNUITY':'AVG_AMT_ANNUITY_CLIENT'})
            return feature_df
        
        else:
            logger.debug('AMT_ANNUITY : column is not present in the DataFrame')

    
    def _extract_avg_credit_client(self):
        ''' Extract features from the 'AMT_CREDIT' in the previous application dataset.

            Features Transformed:
            - AVG_AMT_CREDIT_CLIENT:  average credit amount allocated  per client 

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'AMT_CREDIT' in self.df.columns :
            # AMT_CREDIT means the actual amount of loan the client gets approved
            feature_df = self.df.groupby(by='SK_ID_CURR')['AMT_CREDIT'].mean().to_frame()
            feature_df = feature_df.rename(columns={'AMT_CREDIT':'AVG_AMT_CREDIT_CLIENT'})
            return feature_df
        
        else:
            logger.debug('AMT_CREDIT : column is not present in the DataFrame')


    def _extract_avg_credit_application_ratio(self):
        ''' Extract features from the 'AMT_CREDIT'  and 'AMT_APPLICATION' in the previous application dataset.

            Features Transformed:
            - AVG_CREDIT_APPLICATION_RATIO:  AMT_CREDIT / AMT_APPLICATION and the average of this value per customer

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'AMT_CREDIT' in self.df.columns and 'AMT_APPLICATION' in self.df.columns  :

            self.df['CREDIT_APPLICATION_RATIO'] = self.df['AMT_CREDIT'] / self.df['AMT_APPLICATION']
            feature_df = self.df.groupby(by='SK_ID_CURR')['CREDIT_APPLICATION_RATIO'].mean().to_frame()
            feature_df = feature_df.rename(columns={'CREDIT_APPLICATION_RATIO':'AVG_CREDIT_APPLICATION_RATIO'})
            return feature_df
        
        else:
            logger.debug('AMT_CREDIT and  AMT_APPLICATION : column are not present in the DataFrame')

   

    def _extract_avg_down_payment_rate(self):
        ''' Extract features from the 'RATE_DOWN_PAYMENT' in the previous application dataset.

            Features Transformed:
            - AVG_DOWN_PAYMENT_RATE:  avg downpayment rate for the per client he got

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'AMT_GOODS_PRICE' in self.df.columns :

            feature_df =  self.df.groupby(by='SK_ID_CURR')['RATE_DOWN_PAYMENT'].mean().to_frame()
            feature_df = feature_df.rename(columns={'RATE_DOWN_PAYMENT':'AVG_DOWN_PAYMENT_RATE'})
            
            return feature_df
        
        else:
            logger.debug('RATE_DOWN_PAYMENT  : column are not present in the DataFrame')

    def _extract_avg_goods_price(self):
        ''' Extract features from the 'AMT_GOODS_PRICE' in the previous application dataset.

            Features Transformed:
            - AVG_AMT_GOODS_PRICE:  avg amount of prvious goods that the loan is applied for

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'AMT_GOODS_PRICE' in self.df.columns :

            feature_df = self.df.groupby(by='SK_ID_CURR')['AMT_GOODS_PRICE'].mean().to_frame()
            feature_df = feature_df.rename(columns={'AMT_GOODS_PRICE':'AVG_AMT_GOODS_PRICE'})

            return feature_df
        
        else:
            logger.debug('AMT_GOODS_PRICE  : column are not present in the DataFrame')

    
    def _extract_mean_privileged_interest_rate(self):
        ''' Extract features from the 'RATE_INTEREST_PRIVILEGED' in the previous application dataset.

            Features Transformed:
            
            - AVG_PRIVILEGED_RATE_FLAG:  proportion of applications where the loan interest rate was privileged

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'RATE_INTEREST_PRIVILEGED' in self.df.columns :

            self.df['FLAG_HAD_RATE_INTEREST_PRIVILEGED'] = np.where( self.df['RATE_INTEREST_PRIVILEGED'] > 0 ,
                    1,
                    0)
            feature_df = self.df.groupby(by='SK_ID_CURR')['FLAG_HAD_RATE_INTEREST_PRIVILEGED'].mean().to_frame()
            feature_df = feature_df.rename(columns={'FLAG_HAD_RATE_INTEREST_PRIVILEGED':'AVG_PRIVILEGED_RATE_FLAG'})
            return feature_df
        
        else:
            logger.debug('RATE_INTEREST_PRIVILEGED  : column are not present in the DataFrame')
       

    def _extract_ratio_refused_loans(self):

        ''' Extract features from the 'NAME_CONTRACT_STATUS' in the previous application dataset.

            Features Transformed:
            - RATIO_REFUSED_LOANS: ratio of the refused loans.  refused / approved + refused

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'NAME_CONTRACT_STATUS' in self.df.columns :

            temp = pd.crosstab(self.df['SK_ID_CURR'],self.df['NAME_CONTRACT_STATUS'],dropna=False)
            temp['TOTAL_LOANS']=temp[['Approved','Refused']].sum(axis=1)

            temp['RATIO_REFUSED_LOANS'] = temp['Refused'] / temp['TOTAL_LOANS']
            feature_df = temp['RATIO_REFUSED_LOANS'].fillna(0).to_frame('RATIO_REFUSED_LOANS')
            return feature_df
        
        else:
            logger.debug('NAME_CONTRACT_STATUS  : column are not present in the DataFrame')



    def _extract_num_refused_loans_d(self,days_window):

        ''' Extract features from the 'DAYS_DECISION' and  NAME_CONTRACT_STATUS in the previous application dataset.

            Features Transformed:
            - LOANS_REFUSED_RECENT_D: num of refused loans in the last days_window ex: 60 d 180 d 

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'DAYS_DECISION' in self.df.columns and 'NAME_CONTRACT_STATUS' in self.df.columns :

            filt = ((self.df['DAYS_DECISION'] > -days_window) & (self.df['NAME_CONTRACT_STATUS'] == 'Refused'))

            temp = self.df.loc[filt]

            feature_df = temp.groupby(by='SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame()
            feature_df = feature_df.rename(columns={'SK_ID_PREV':f'LOANS_REFUSED_RECENT_{days_window}D'})


            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index,fill_value=0)
            
            return feature_df
        
        else:
            logger.debug('NAME_CONTRACT_STATUS and DAYS_DECISION : column are not present in the DataFrame')

    def _extract_flag_hc_reject_loans_2y(self):

        ''' Extract features from the 'DAYS_DECISION' and  CODE_REJECT_REASON in the previous application dataset.

            Features Transformed:
            - FLAG_HC_REJECT_REASON_2Y: Flag if the client has the loan  rejected loan due to the HC reject reason in last 2 year
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'CODE_REJECT_REASON' in self.df.columns and 'DAYS_DECISION' in self.df.columns:

            
            filt = (self.df['DAYS_DECISION'] > -730)

            temp = self.df.loc[filt].copy()

            temp['FLAG_HC_REJECT_REASON_2Y'] = np.where(temp['CODE_REJECT_REASON'] =='HC',
                                       1,
                                       0)

            feature_df = temp.groupby(by='SK_ID_CURR')['FLAG_HC_REJECT_REASON_2Y'].max().to_frame()
            
            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index,fill_value=0)
            
                        
            return feature_df
        
        else:
            logger.debug('CODE_REJECT_REASON : column are not present in the DataFrame')

    def _extract_num_sco_scofr_reject_loans(self):

        ''' Extract features from the CODE_REJECT_REASON in the previous application dataset.

            Features Transformed:
            - NUM_SCO_SCOFR_REJECT_REASON: total num of loan rejected for the SCO or SCOFR reason per client. 
            
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'CODE_REJECT_REASON' in self.df.columns :

            self.df['NUM_SCO_SCOFR_REJECT_REASON'] = np.where(self.df['CODE_REJECT_REASON'].isin(['SCO','SCOFR']),
                                       1,
                                       0)
           
            feature_df =self.df.groupby(by='SK_ID_CURR')['NUM_SCO_SCOFR_REJECT_REASON'].sum().to_frame()
            return feature_df
        
        else:
            logger.debug('CODE_REJECT_REASON : column are not present in the DataFrame')


    def _extract_num_alone_applications(self):

        ''' Extract features from the NAME_TYPE_SUITE in the previous application dataset.

            Features Transformed:
            - NUM_ALONE_APPLICATIONS: total num of loan where client was alone to apply for the loan
            
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NAME_TYPE_SUITE' in self.df.columns :

            self.df['FLAG_IS_ALONE'] = (self.df['NAME_TYPE_SUITE'] == 'Unaccompanied').astype(int)

            feature_df = self.df.groupby(by='SK_ID_CURR')['FLAG_IS_ALONE'].sum().to_frame()

            feature_df = feature_df.rename(columns={'FLAG_IS_ALONE':'NUM_ALONE_APPLICATIONS'})
            return feature_df
        
        else:
            logger.debug('NAME_TYPE_SUITE : column are not present in the DataFrame')

    

    def _extract_repeater_new_client_flag_1y(self):

        ''' Extract features from the NAME_CLIENT_TYPE in the previous application dataset.

            Features Transformed:
            - FLAG_REPEATER: if the client is repeater in last 1 year 
            - FLAG_NEW_CLIENT:  if the client is new in last 1 year
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NAME_CLIENT_TYPE' in self.df.columns :

            self.df['FLAG_REPEATER'] = (self.df['NAME_CLIENT_TYPE'] == 'Repeater').astype(int)

            self.df['FLAG_NEW_CLIENT'] = ((self.df['DAYS_DECISION'] > -365) &  (self.df['NAME_CLIENT_TYPE'] == 'New')).astype(int)

            feature_df = self.df.groupby('SK_ID_CURR')[['FLAG_REPEATER', 'FLAG_NEW_CLIENT']].max()

            return feature_df
        
        else:
            logger.debug('NAME_CLIENT_TYPE : column are not present in the DataFrame')
    

    def _extract_num_cash_loans_6m(self):

        ''' Extract features from the NAME_PORTFOLIO and DAYS_DECISION in the previous application dataset.

            Features Transformed:
            - NUM_CASH_LOANS_RECENT_180D: num cash loans in last 180 Days

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NAME_PORTFOLIO' in self.df.columns and 'DAYS_DECISION' in self.df.columns:

            filt_6m_cash_loans = self.df[((self.df['NAME_PORTFOLIO'] == 'Cash') & (self.df['DAYS_DECISION'] > -180))]

            feature_df = filt_6m_cash_loans.groupby(by='SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame()
            feature_df = feature_df.rename(columns={'SK_ID_PREV':'NUM_CASH_LOANS_RECENT_180D'})

            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index,fill_value=0)

            return feature_df
        
        else:
            logger.debug('DAYS_DECISION and NAME_PORTFOLIO : column are not present in the DataFrame')


    def _extract_num_high_channel_method(self):

        ''' Extract features from the CHANNEL_TYPE  in the previous application dataset.

            Features Transformed:
            - NUM_HIGH_RISK_CHANNEL_METHOD:  num of high risk channel method where client apply for the loan 

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'CHANNEL_TYPE' in self.df.columns:

            self.df['NUM_HIGH_RISK_CHANNEL_METHOD'] = self.df['CHANNEL_TYPE'].isin(['AP+ (Cash loan)','Contact center']).astype(int)

            feature_df = self.df.groupby(by='SK_ID_CURR')['NUM_HIGH_RISK_CHANNEL_METHOD'].sum().to_frame()

            return feature_df
        
        else:
            logger.debug('CHANNEL_TYPE  : column are not present in the DataFrame')


    def _extract_ratio_insured_loans(self):

        ''' Extract features from the NFLAG_INSURED_ON_APPROVAL  in the previous application dataset.

            Features Transformed:
            - RATIO_INSURED_LOANS: ratio of the insured loans of the client

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NFLAG_INSURED_ON_APPROVAL' in self.df.columns:
            
            feature_df = self.df.groupby('SK_ID_CURR')['NFLAG_INSURED_ON_APPROVAL'].mean().to_frame('RATIO_INSURED_LOANS')

            return feature_df
        
        else:
            logger.debug('NFLAG_INSURED_ON_APPROVAL  : column are not present in the DataFrame')


    def _extract_ratio_highrisk_yield_loans(self):

        ''' Extract features from the NAME_YIELD_GROUP  in the previous application dataset.

            Features Transformed:
            - RATIO_HIGH_RISK_YIELD_LOANS: ratio of high risk yeild loans to total loans

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NAME_YIELD_GROUP' in self.df.columns:

            self.df['HIGH_RISK_YIELD_LOANS'] = np.where(self.df['NAME_YIELD_GROUP']=='high',1,0)
            feature_df = self.df.groupby('SK_ID_CURR')['HIGH_RISK_YIELD_LOANS'].mean().to_frame('RATIO_HIGH_RISK_YIELD_LOANS')

            return feature_df
        
        else:
            logger.debug('NAME_YIELD_GROUP  : column are not present in the DataFrame')


    def _extract_avg_max_loan_delay(self):

        ''' Extract features from the DAYS_LAST_DUE and  DAYS_TERMINATION in the previous application dataset.

            Features Transformed:
            - MEAN_LOAN_REPAYMENT_DIFF: mean of loan repayment differences per client across all loans.
            - MAX_LOAN_REPAYMENT_DIFF: max of loan repayment differences per client across all loans.

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'DAYS_TERMINATION' in self.df.columns and  'DAYS_LAST_DUE' in self.df.columns:

            self.df['DAYS_TERMINATION'] = self.df['DAYS_TERMINATION'].replace({365243.0:np.nan})
            self.df['DAYS_LAST_DUE'] = self.df['DAYS_LAST_DUE'].replace({365243.0:np.nan})
            self.df['LOAN_REPAYMENT_DIFF'] = self.df['DAYS_TERMINATION'] - self.df['DAYS_LAST_DUE']
            feature_df = self.df.groupby('SK_ID_CURR')['LOAN_REPAYMENT_DIFF'].agg(MEAN_LOAN_REPAYMENT_DIFF='mean', MAX_LOAN_REPAYMENT_DIFF='max') 
            return feature_df
        
        else:
            logger.debug('DAYS_TERMINATION and  DAYS_LAST_DUE : column are not present in the DataFrame')

     
 
    def add_features_main(self,main_df):

        '''Extract and Create feature from Previous Application dataset and append in Main Dataframe'''   
        try:
            self.feature_extractors  = [
                self._extract_avg_amt_annuity,
                self._extract_avg_credit_client,
                self._extract_avg_credit_application_ratio,
                self._extract_avg_down_payment_rate,
                self._extract_avg_goods_price,
                self._extract_mean_privileged_interest_rate,
                self._extract_ratio_refused_loans,
                lambda : self._extract_num_refused_loans_d(60),
                lambda : self._extract_num_refused_loans_d(180),
                self._extract_flag_hc_reject_loans_2y,
                self._extract_num_sco_scofr_reject_loans,
                self._extract_num_alone_applications,
                self._extract_repeater_new_client_flag_1y,
                self._extract_num_cash_loans_6m,
                self._extract_num_high_channel_method,
                self._extract_ratio_insured_loans,
                self._extract_ratio_highrisk_yield_loans,
                self._extract_avg_max_loan_delay
            ]

            for extractor in self.feature_extractors:
                features_df = extractor()
                main_df = main_df.merge(features_df,on='SK_ID_CURR',how='left')

            logger.info("Aggregated features from Previous Application dataset dataframe successfully merged into the main  Application dataframe.")

            return main_df
         
        except Exception as e:
            raise MyException(e,sys,logger)






class CreditBalanceTransformation(BaseTransformer):
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_config: DataIngestionConfig):

        super().__init__(data_transformation_config, data_ingestion_config)

        self.df = self.load_data(
            self.data_ingestion_config.credit_card_balance,
            self.data_transformation_config.credit_card_balance_reduce_dtypes
            )

    def _extract_ratio_credit_utlization_1m(self):

        ''' Extract features from the MONTHS_BALANCE and  AMT_CREDIT_LIMIT_ACTUAL and AMT_BALANCE in the Credit Balance dataset.

            Features Transformed:
            - RATIO_CREDIT_UTLIZATION_DEBT_1M: Average ratio utilization per customer across theire active lines
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE'}
        if required_cols.issubset(self.df.columns):
            # Produces an average utilization per customer across their active lines.

            filt_1m_df = self.df[self.df['MONTHS_BALANCE'] >= -1].copy()

            filt_1m_df['RATIO_CREDIT_UTLIZATION_DEBT_1M'] = np.where(
                (filt_1m_df['MONTHS_BALANCE'] == -1) & (filt_1m_df['AMT_CREDIT_LIMIT_ACTUAL'] > 0),
                filt_1m_df['AMT_BALANCE'] / filt_1m_df['AMT_CREDIT_LIMIT_ACTUAL'],
                np.nan
            )


            
            feature_df = filt_1m_df.groupby('SK_ID_CURR')['RATIO_CREDIT_UTLIZATION_DEBT_1M'].mean().to_frame('MEAN_RATIO_CREDIT_UTILIZATION_DEBT_1M')

            # Ensure all customers present
            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index, fill_value=0)
            
            return feature_df
        
        else:
            logger.debug("'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE' column are not present in the DataFrame")

         
    def _extract_avg_credit_usage_12m(self):

        ''' Extract features from the MONTHS_BALANCE and  AMT_DRAWINGS_CURRENT AND AMT_CREDIT_LIMIT_ACTUAL in the Credit Balance dataset.

            Features Transformed:
            - CREDIT_USAGE_RATIO_12M_ACTIVE_MEAN:  Average credit usage ratio over the last 12 months
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT', 'AMT_CREDIT_LIMIT_ACTUAL'}
        if required_cols.issubset(self.df.columns):

            filt_12m_df = self.df[self.df['MONTHS_BALANCE'] >= -12].copy()


            filt_12m_df['CREDIT_USAGE_RATIO_PER_MONTH'] = np.where((filt_12m_df['MONTHS_BALANCE'] >= -12) & (filt_12m_df['AMT_CREDIT_LIMIT_ACTUAL'] > 0),
                                                    filt_12m_df['AMT_DRAWINGS_CURRENT'] / filt_12m_df['AMT_CREDIT_LIMIT_ACTUAL'],
                                                    np.nan)

            feature_df = filt_12m_df.groupby(by='SK_ID_CURR')['CREDIT_USAGE_RATIO_PER_MONTH'].mean().to_frame('CREDIT_USAGE_RATIO_12M_ACTIVE_MEAN')
            feature_df['CREDIT_USAGE_RATIO_12M_ACTIVE_MEAN'] = feature_df['CREDIT_USAGE_RATIO_12M_ACTIVE_MEAN'].fillna(0)

            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index, fill_value=0)
          
            return feature_df
        else:
            logger.debug("'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL',AMT_DRAWINGS_CURRENT  column are not present in the DataFrame")
     
     
    def _extract_avg_atm_drawings_6m(self):

        ''' Extract features from the MONTHS_BALANCE and  AMT_DRAWINGS_ATM_CURRENT in the Credit Balance dataset.

            Features Transformed:
            - AVG_ATM_DRAWINGS_RECENT_6M: Average atm drawing over the last 6 months
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT',}

        if required_cols.issubset(self.df.columns):

            filt_6m_df = self.df[self.df['MONTHS_BALANCE'] >= -6].copy()
            feature_df=  filt_6m_df.groupby(by='SK_ID_CURR')['AMT_DRAWINGS_ATM_CURRENT'].mean().to_frame('AVG_ATM_DRAWINGS_RECENT_6M')


            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index, fill_value=0)
          
            return feature_df
        
        else:
            logger.debug("'MONTHS_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT',  column are not present in the DataFrame")


    def _extract_avg_pos_spend_ratio_1y(self):

        ''' Extract features from the MONTHS_BALANCE and  AMT_DRAWINGS_POS_CURRENT and AMT_CREDIT_LIMIT_ACTUAL in the Credit Balance dataset.

            Features Transformed:
            - POS_SPEND_RATIO_PER_MONTH: Average of ratio of POS spending / credit limit  over the last 12 months
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_DRAWINGS_POS_CURRENT','AMT_CREDIT_LIMIT_ACTUAL'}
        if required_cols.issubset(self.df.columns):
            
            filt_12m_df = self.df[self.df['MONTHS_BALANCE'] >= -12].copy()


            filt_12m_df['POS_SPEND_RATIO_PER_MONTH'] = np.where( filt_12m_df['AMT_CREDIT_LIMIT_ACTUAL'] > 0,
                                               filt_12m_df['AMT_DRAWINGS_POS_CURRENT'] /filt_12m_df['AMT_CREDIT_LIMIT_ACTUAL'],
                                                np.nan)

            feature_df = filt_12m_df.groupby(by='SK_ID_CURR')['POS_SPEND_RATIO_PER_MONTH'].mean().to_frame('POS_SPEND_RATIO_MEAN_1Y')
            
            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index, fill_value=0)
        
            return feature_df
        
        else:
            logger.debug(" MONTHS_BALANCE', 'AMT_DRAWINGS_POS_CURRENT','AMT_CREDIT_LIMIT_ACTUAL' column are not present in the DataFrame")
         

    def _extract_avg_atm_withdraws_credit_6m(self):

        ''' Extract features from the MONTHS_BALANCE and  AMT_DRAWINGS_ATM_CURRENT and AMT_CREDIT_LIMIT_ACTUAL in the Credit Balance dataset.

            Features Transformed:
            - ATM_WITHDRAWAL_RATIO_MEAN_6M: Average of ratio of atm drawings / credit limit  over the last 6 months
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT','AMT_CREDIT_LIMIT_ACTUAL'}
        if required_cols.issubset(self.df.columns):
            
            filt_6m_df = self.df[self.df['MONTHS_BALANCE'] >= -6].copy()


            filt_6m_df['ATM_WITHDRAWAL_RATIO'] = np.where(
                filt_6m_df['AMT_CREDIT_LIMIT_ACTUAL'] > 0,
                filt_6m_df['AMT_DRAWINGS_ATM_CURRENT'] /filt_6m_df['AMT_CREDIT_LIMIT_ACTUAL'],
                np.nan
                )

            feature_df = filt_6m_df.groupby('SK_ID_CURR')['ATM_WITHDRAWAL_RATIO'].mean().to_frame('ATM_WITHDRAWAL_RATIO_MEAN_6M')

            feature_df['ATM_WITHDRAWAL_RATIO_MEAN_6M'] = feature_df['ATM_WITHDRAWAL_RATIO_MEAN_6M'].fillna(0)

            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index, fill_value=0)
        
            return feature_df
        
        else:
            logger.debug(" MONTHS_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT','AMT_CREDIT_LIMIT_ACTUAL' column are not present in the DataFrame")
        


    def _extract_payment_ratio_coverage_6m(self):

        ''' Extract features from the MONTHS_BALANCE and  AMT_INST_MIN_REGULARITY and  AMT_PAYMENT_CURRENT  in the Credit Balance dataset.

            Features Transformed:
            - LATEST_PAYMENT_COVERAGE_RATIO_6M:  Average ratio of payments made to minimum required payments over the last 6 months.
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_INST_MIN_REGULARITY','AMT_PAYMENT_CURRENT'}
        if required_cols.issubset(self.df.columns):

            filt_6m_df = self.df[self.df['MONTHS_BALANCE'] >= -6].copy()
            
             # the ratio of recent 6 months  mean  for repayment behaviour
            filt_6m_df['LATEST_PAYMENT_COVERAGE_RATIO_6M'] = np.where(
                filt_6m_df['AMT_INST_MIN_REGULARITY'] > 0,
                filt_6m_df['AMT_PAYMENT_CURRENT'] / filt_6m_df['AMT_INST_MIN_REGULARITY'],
                np.nan
            )

            feature_df=  filt_6m_df.groupby(by='SK_ID_CURR')['LATEST_PAYMENT_COVERAGE_RATIO_6M'].mean().to_frame('AVG_LATEST_PAYMENT_COVERAGE_RATIO_6M')


            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index, fill_value=0)
        
            return feature_df
        
        else:
            logger.debug(" MONTHS_BALANCE', 'AMT_INST_MIN_REGULARITY','AMT_PAYMENT_CURRENT' column are not present in the DataFrame")
        


    def _extract_payment_ratio_coverage_overall(self):

        ''' Extract features from the MONTHS_BALANCE and  AMT_INST_MIN_REGULARITY and  AMT_PAYMENT_CURRENT and AMT_CREDIT_LIMIT_ACTUAL in the Credit Balance dataset.

            Features Transformed:
            - PAYMENT_COVERAGE_RATIO: Mean repayment coverage ratio per client, average of payments  to expected installments.
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_INST_MIN_REGULARITY','AMT_PAYMENT_CURRENT'}
        if required_cols.issubset(self.df.columns):
            #overall repayment discipline of the clients
            self.df['PAYMENT_COVERAGE_RATIO'] = np.where(
                self.df['AMT_INST_MIN_REGULARITY'] > 0,
                self.df['AMT_PAYMENT_CURRENT'] / self.df['AMT_INST_MIN_REGULARITY'],
                np.nan
            )
            feature_df=  self.df.groupby(by='SK_ID_CURR')['PAYMENT_COVERAGE_RATIO'].mean().to_frame('AVG_PAYMENT_COVERAGE_RATIO')

        

            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index, fill_value=0)
        
            return feature_df
        
        else:
            logger.debug(" MONTHS_BALANCE', 'AMT_INST_MIN_REGULARITY','AMT_PAYMENT_CURRENT' column are not present in the DataFrame")
        
    def _extract_credit_utilization_ratios(self):

        ''' Extract features from the, 'AMT_RECEIVABLE_PRINCIPAL','AMT_CREDIT_LIMIT_ACTUAL','AMT_TOTAL_RECEIVABLE' in the Credit Balance dataset.

            Features Transformed:
            - AVG_PRINCIPAL_RATIO: Average ratio of receivable principal to credit limit.
            - AVG_TOTAL_RECEIVABLE_RATIO: Average ratio of total receivables (principal + interest + fees)  to credit limit.
           
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = { 'AMT_RECEIVABLE_PRINCIPAL','AMT_CREDIT_LIMIT_ACTUAL','AMT_TOTAL_RECEIVABLE'}
        if required_cols.issubset(self.df.columns):
            
            #represents how much principal amount of their credit limit is currently used.

            self.df['PRINCIPAL_RATIO'] = np.where(
                self.df['AMT_CREDIT_LIMIT_ACTUAL'] > 0,
                self.df['AMT_RECEIVABLE_PRINCIPAL'] / self.df['AMT_CREDIT_LIMIT_ACTUAL'],
                np.nan
                )

            #represents how much of their total credit limit is currently used (principal + interest + fees).
            self.df['TOTAL_RECEIVABLE_RATIO'] = np.where(
                self.df['AMT_CREDIT_LIMIT_ACTUAL'] > 0,
                self.df['AMT_TOTAL_RECEIVABLE'] / self.df['AMT_CREDIT_LIMIT_ACTUAL'],
                np.nan
            )

            feature_df = self.df.groupby('SK_ID_CURR')[['PRINCIPAL_RATIO', 'TOTAL_RECEIVABLE_RATIO']].mean()
            feature_df.columns = ['AVG_PRINCIPAL_RATIO', 'AVG_TOTAL_RECEIVABLE_RATIO']
            
            # Reindex to include all customers

            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index, fill_value=0)
        
            return feature_df
        
        else:
            logger.debug(" 'AMT_RECEIVABLE_PRINCIPAL','AMT_CREDIT_LIMIT_ACTUAL','AMT_TOTAL_RECEIVABLE' column are not present in the DataFrame")
        
    def _extract_cnt_installments_mean(self):

        ''' Extract features from CNT_INSTALMENT_MATURE_CUM the in the Credit Balance dataset.

            Features Transformed:
            - AVG_INSTALLMENTS_PER_CUST: the average number of installments that a customer has paid across all their previous credit cards or loans.

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'CNT_INSTALMENT_MATURE_CUM'}
        if required_cols.issubset(self.df.columns):
            
            feature_df  = self.df.groupby(by='SK_ID_CURR')['CNT_INSTALMENT_MATURE_CUM'].mean().to_frame('AVG_INSTALLMENTS_PER_CUST')

            
            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            feature_df = feature_df.reindex(all_cust_index, fill_value=0)
        
            return feature_df
        
        else:
            logger.debug(" CNT_INSTALMENT_MATURE_CUM column are not present in the DataFrame")

    def add_features_main(self,main_df):

        '''Extract and Create feature from Credit card balance dataset and append in Main Dataframe'''   
        try:
            self.feature_extractors  = [
                self._extract_ratio_credit_utlization_1m,
                self._extract_avg_credit_usage_12m,
                self._extract_avg_atm_drawings_6m,
                self._extract_avg_pos_spend_ratio_1y,
                self._extract_avg_atm_withdraws_credit_6m,
                self._extract_payment_ratio_coverage_6m,
                self._extract_payment_ratio_coverage_overall,
                self._extract_credit_utilization_ratios,
                self._extract_cnt_installments_mean
            ]

            for extractor in self.feature_extractors:
                features_df = extractor()
                main_df = main_df.merge(features_df,on='SK_ID_CURR',how='left')

            logger.info("Aggregated features from Credit card balance dataframe successfully merged into the main  Application dataframe.")

            return main_df
         
        except Exception as e:
            raise MyException(e,sys,logger)

      
class DataTransformation:
     
    def __init__(self,data_transformation_config:DataTransformationConfig,data_ingestion_config:DataIngestionConfig):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_config = data_ingestion_config 

    def _is_data_validated(self):
        '''check and load status of the data validation from yaml file
            
            return:
                True | False : status of data validation
        '''

        file = read_yaml_file(self.data_transformation_config.data_validation_yaml,logger)
        return file['is_data_validated']

    def run(self):
        if self._is_data_validated():
            logger.info('DATA IS VALIDATED')
            try:
                main_df_path =  os.path.join(self.data_ingestion_config.artifact_raw_dir,r'application_data.csv')

                main_transformer = ApplicationDfTransformer(main_df_path,ApplicationDfConfig())

            
          
                # after preprocessing the main_df
                main_df =  main_transformer.run_preprocessing_steps()
                
                
                # list of all the classes of transforamtion
                classes = [BureauBalanceTransformation,
                           BureauTransformer,
                           InstallmentsPaymentsTransformation,
                           PosCashBalanceTransformation,
                           PreviousApplicationsTransformation,
                           CreditBalanceTransformation]
                
                for cls in classes:
                    obj = cls(self.data_transformation_config, self.data_ingestion_config)
                    main_df =  obj.add_features_main(main_df)

                main_df_transformed = main_df.copy()                   
            
                logger.info('DATA TRANSFORMATION DONE SUCCESSFULLY')
               
                main_df_transformed_path = os.path.join(self.data_ingestion_config.artifact_interim_dir,'main_df_transformed.csv')                
                os.makedirs(self.data_ingestion_config.artifact_interim_dir,exist_ok=True)
                main_df_transformed.to_csv(main_df_transformed_path,index=False)

                logger.info(f'Main Df Transformed is saved here: {main_df_transformed_path}')


            except Exception as e:
                raise MyException(e,sys,logger)
        else:
            logger.error(f'Data Is Not Validated')
 
if __name__ =='__main__':
    
    data_transformation_config = DataTransformationConfig()
    data_ingestion_config = DataIngestionConfig()

    data_transformation = DataTransformation(
        data_transformation_config=data_transformation_config,
        data_ingestion_config=data_ingestion_config
    )

    data_transformation.run()


   
