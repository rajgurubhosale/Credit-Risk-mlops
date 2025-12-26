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
import gc
from src.entity.data_ingestion_artifact import * 
from src.entity.data_transformation_artifact import *
from src.entity.data_ingestion_artifact import *
from src.entity.data_validation_artifact import *
from src.constants.data_transformation_constant import *

PLACEHOLDER = - 99999

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
        
    def _replace_placeholders(self):
        '''
        Convert categorical placeholders to Missing and numerical to -99999  PLACEHOLDER to have uniformility

        placeholder_values :
            - XNA / XAP / UNknown: Missing
            - 365243, -1000.67 : -99999 PLACEHOLDER

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
        
    def run_preprocessing_steps(self):

        '''run all preprocessing steps in sequence

            return:
                self.main_df: returns the dataframe after all preprocessing
            '''
        try:
            self._preprocessing()
            self._convert_days_to_years()
            self._replace_placeholders()

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
        dpd_map = {'X':np.nan,'C':0,'1':1,'2':2,'3':3,'4':4,'5':5}
        self.bureau_balance['STATUS'] = self.bureau_balance['STATUS'].map(dpd_map)
        self.new_features_cols = []
    def _safe_join(self,base,new)-> pd.DataFrame:
        ''' helper function'''
        if base.empty:
            return new
        else:
            return base.join(new, how='outer')

    def _extract_worst_dpd_features(self):
        '''  Create worst DPD (Days Past Due) features for multiple time frames 
            from the bureau_balance dataset.
    
            Features Extracted:
            - WORST_DPD_ based on time frame:[3, 6, 9, 12, 24, 36, 72, 96] M

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as index and WORST_DPD_BUREAU_XM features
                Missing values filled with the placeholder -99999

        '''
        if 'STATUS' in self.bureau_balance.columns:

            time_frames = [3, 6, 9, 12, 24, 36, 72, 96]
            # empty dataframe to apend the feature into
            features = []

            for frame in time_frames:
                filt = self.bureau_balance['MONTHS_BALANCE'] >= -frame
                temp = self.bureau_balance.loc[filt].copy()
                feature_dpd = temp.groupby(by='SK_ID_BUREAU')['STATUS'].max().to_frame(f'WORST_DPD_BUREAU_{frame}M')
                features.append(feature_dpd)

            features = pd.concat(features, axis=1).reset_index()

            bureau_agg = (
                self.bureau[['SK_ID_BUREAU', 'SK_ID_CURR']]
                .merge(features, on='SK_ID_BUREAU', how='left')
                .groupby('SK_ID_CURR')
                .max()
                .reset_index()
                )
            
            bureau_agg = bureau_agg.drop(columns='SK_ID_BUREAU')
            return bureau_agg


        else:
            logger.debug('STATUS : column is not present in the DataFrame')
           
  
    def _extract_severe_dpd_features(self):
        '''  Create Severe DPD (Days Past Due) i.e dpd>=3 features for multiple time frames 
            from the bureau_balance dataset.
    
            Features Extracted:
            - SEVERE_DPD_ based on time frame:[12,24,36,72]M

            Returns:
            - feature_df : 
                DataFrame with SK_ID_BUREAU as index and SEVERE_DPD_BUREAU_XM features
                Missing values filled with the placeholder -99999

        '''
        if 'STATUS' in self.bureau_balance.columns:
                        
            time_frames = [12,24,36,72]
            # empty dataframe to apend the feature into
            features = []

            for frame in time_frames:
                filt = (self.bureau_balance['MONTHS_BALANCE'] >= -frame) & (self.bureau_balance['STATUS'] >= 3)
                temp = self.bureau_balance.loc[filt].copy()
                feature_dpd = temp.groupby(by='SK_ID_BUREAU')['STATUS'].max().to_frame(f'SEVERE_DPD_BUREAU_{frame}M')
                features.append(feature_dpd)

            features = pd.concat(features, axis=1).reset_index()

            bureau_agg = (
                self.bureau[['SK_ID_BUREAU', 'SK_ID_CURR']]
                .merge(features, on='SK_ID_BUREAU', how='left')
                .groupby('SK_ID_CURR')
                .max()
                .reset_index()
                )
            bureau_agg = bureau_agg.drop(columns='SK_ID_BUREAU')

            return bureau_agg
        else:
            logger.debug('STATUS : column is not present in the DataFrame')
             
    def _extract_month_recent_dpd(self):
        '''  Extract the most recent month of the where dpd > 0. 
    
            Features Extracted:
            - RECENT_MONTH_OF_DPD: Most recent Month which have  DPD

            Returns:
            - feature_df : 
                DataFrame with SK_ID_BUREAU as index and RECENT_MONTH_OF_DPD feature
                Missing values filled with the placeholder -99999

        '''
        if 'STATUS' in self.bureau_balance.columns:
                        
            temp = self.bureau_balance.loc[self.bureau_balance['STATUS'] > 0]

            features = (temp.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max().to_frame('RECENT_MONTH_OF_DPD').reset_index())

            bureau_agg = (
                self.bureau[['SK_ID_BUREAU', 'SK_ID_CURR']]
                .merge(features, on='SK_ID_BUREAU', how='left')
                .groupby('SK_ID_CURR')
                .max()
                .reset_index()
            )
            bureau_agg = bureau_agg.drop(columns='SK_ID_BUREAU')

            return bureau_agg
        else:
            logger.debug('STATUS : column is not present in the DataFrame')
            
    def add_features_main(self, main_df):
        """Extract and attach bureau-balance features to main_df"""
        try:
            features_extractors = [
                self._extract_worst_dpd_features,
                self._extract_severe_dpd_features,
                self._extract_month_recent_dpd
            ]

            for extractor in features_extractors:
                features_df = extractor()


                main_df = main_df.merge(
                    features_df,
                    on='SK_ID_CURR',
                    how='left'
                )
                # log the method is running
                method_name = extractor.__name__
                logger.info(f"Running {self.__class__.__name__}.{method_name}")

                new_cols = features_df.columns.drop('SK_ID_CURR')

                main_df[new_cols] = main_df[new_cols].fillna(PLACEHOLDER)

            logger.info(
                "Successfully merged Bureau Balance features into main dataframe."
            )

            return main_df

        except Exception as e:
            raise MyException(e, sys, logger)



class BureauTransformer(BaseTransformer):
    '''Extracts and Transform features from the bureau data. And append in the main dataframe'''
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_config: DataIngestionConfig):
        '''Load the bureau dataset and save in the self.df dataframe'''
        super().__init__(data_transformation_config, data_ingestion_config)

        self.df = self.load_data(
            data_path = self.data_ingestion_config.bureau_data,
            dtypes = self.data_transformation_config.bureau_dtypes_reduce
        )
    
    def _extract_num_credit_currencies(self):

        '''  Count the number of different currencies a client has taken loans in.

            Features Transformed:
            - NUM_CREDIT_CURRENCIES: number of unique credit currencies the client had taken the loan

            Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''
        if 'CREDIT_CURRENCY' in self.df.columns:
            feature_df = self.df.groupby(by='SK_ID_CURR')['CREDIT_CURRENCY'].nunique().to_frame('NUM_CREDIT_CURRENCIES')

            feature_df = feature_df.fillna(0)
            return feature_df.reset_index()

        else:
            logger.debug('CREDIT_CURRENCY : column is not present in the DataFrame')

    def _extract_num_active_credit_d(self):
        '''  Create num active credit features for multiple time frames 
            from the bureau dataset.
    
            Features Extracted:
            - NUM_ACTIVE_CREDIT_XD: based on time frame: [90, 180, 270, 360, 720] Days
        

            Returns:
            - feature_df : 
                DataFrame with SK_ID_BUREAU as index and NUM_ACTIVE_CREDIT_XD features
                Missing values filled with the placeholder -99999

        '''
        if 'CREDIT_ACTIVE' in self.df .columns:
            

            
            time_frames = [90, 180, 270, 360, 720]

            # empty dataframe to apend the feature into
            features_df = pd.DataFrame()


            for frame in time_frames:
                filt = self.df['DAYS_CREDIT'] >= -frame
                temp = self.df.loc[filt].copy()

                crosstab = pd.crosstab(temp['SK_ID_CURR'],temp['CREDIT_ACTIVE'])
                    

                active_credit = crosstab['Active'].to_frame().rename(columns={'Active':f'NUM_ACTIVE_CREDIT_{frame}D'})

                if features_df.empty:
                    features_df = active_credit.copy()
                else:
                    features_df = features_df.join(active_credit,how='outer')
            features_df = features_df.fillna(0)

            return features_df.reset_index()
        else:
            logger.debug('STATUS : column is not present in the DataFrame')


    def _extract_days_last_bad_loan(self):

        '''    Extracts the recency of a bad loan for each customer from the bureau dataset.
            
            Features Transformed:
            - DAYS_SINCE_LAST_BAD_LOAN : days(positive number ) recent bad loan else 0 for the person dont have bad loan
       
            Returns:
                 DataFrame with SK_ID_BUREAU as index and DAYS_SINCE_LAST_BAD_LOAN feature
                    client with no bad loan is filled with values 0
        '''
        if 'CREDIT_ACTIVE' in self.df.columns and 'DAYS_CREDIT' in self.df.columns:

   
            filt = self.df['CREDIT_ACTIVE'].isin(['Bad debt','Sold'])
            temp = self.df.loc[filt].copy()
            features_df = temp.groupby(by='SK_ID_CURR')['DAYS_CREDIT'].max().to_frame('DAYS_SINCE_LAST_BAD_LOAN')
            
            # converting it into positive number so i can use the -99999 placeholder later for the null values
            features_df['DAYS_SINCE_LAST_BAD_LOAN'] = -features_df['DAYS_SINCE_LAST_BAD_LOAN']


            return features_df.reset_index()

        else:
            logger.debug('CREDIT_ACTIVE and DAYS_CREDIT : column is not present in the DataFrame')


       
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

            return features_df.reset_index()
        else:
            logger.debug('CREDIT_DAY_OVERDUE : column is not present in the DataFrame')


    def _extract_days_enddate(self):

        ''' Extract features from the 'DAYS_ENDDATE_FACT' and 'DAYS_CREDIT_ENDDATE' column in the bureau dataframe.
            
            Features Transformed:
            - AVG_REPAYMENT_DAYS_DIFF : Average diff in days between actual and scheduled credit end date for closed credits.
            - MAX_REPAYMENT_DAYS_DIFF : Max diff in days between actual and scheduled credit end date for closed credits.
           
             Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''
                
        if ('DAYS_ENDDATE_FACT' in self.df.columns) and ('DAYS_CREDIT_ENDDATE' in self.df.columns):

            filt = self.df['CREDIT_ACTIVE'] == 'Closed'
            new_df = self.df.loc[filt].copy()
            new_df['REPAYMENT_DAYS_DIFF'] = (new_df['DAYS_ENDDATE_FACT'] - new_df['DAYS_CREDIT_ENDDATE'])
            
            features_df = new_df.groupby(by='SK_ID_CURR')['REPAYMENT_DAYS_DIFF'].agg(['mean','max']).rename(columns={"mean":"AVG_REPAYMENT_DAYS_DIFF",
                 "max":"MAX_REPAYMENT_DAYS_DIFF"})
                

            return features_df.reset_index()
        else:
            logger.debug('DAYS_ENDDATE_FACT, DAYS_CREDIT_ENDDATE: column is not present in the dataframe')


    def _amt_credit_max_overdue(self):

        ''' Extract flag has overdue and max amount overdue features from the 'AMT_CREDIT_MAX_OVERDUE' column
             in the bureau dataframe.
        
            Features Transformed:
            - FLAG_HAS_AMT_OVERDUE: flag if the customer had any amt overdue 1, 0
            - MAX_AMT_OVERDUE: maximum amount overdue of the customer across loans.

            Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''

        if 'AMT_CREDIT_MAX_OVERDUE' in self.df.columns:      

            temp = self.df[['SK_ID_CURR', 'AMT_CREDIT_MAX_OVERDUE']].copy()
  
            temp['FLAG_HAS_AMT_OVERDUE'] = (temp['AMT_CREDIT_MAX_OVERDUE'] > 0 ).astype(int)
            
            filt = temp['AMT_CREDIT_MAX_OVERDUE'].isnull()
            temp.loc[filt,'FLAG_HAS_AMT_OVERDUE'] = np.nan
            
            features_df = temp.groupby(by='SK_ID_CURR')['FLAG_HAS_AMT_OVERDUE'].max().to_frame()
            
            max_overdue = temp.groupby(by='SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].max().to_frame('MAX_AMT_OVERDUE')
            features_df = features_df.join(max_overdue,how='outer')
            return features_df.reset_index()
        else:
            logger.debug('AMT_CREDIT_MAX_OVERDUE: column is not present in the DataFrame')


    def _cnt_credit_prolong(self):
        ''' Extract features from the 'CNT_CREDIT_PROLONG' column in the bureau dataframe.
    
            Features Transformed:
            - FLAG_HAS_CREDIT_PROLONG: flag if the customer credit prolong
            - MAX_CREDIT_PROLONG: max credit prolong for the customer

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
            max_prolong = self.df.groupby(by='SK_ID_CURR')['CNT_CREDIT_PROLONG'].max().to_frame('MAX_CREDIT_PROLONG')
            features_df = features_df.join(max_prolong,how='outer')
            return features_df.reset_index()
           
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

            return features_df.reset_index()

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

            return features_df.reset_index()
        else:
            logger.debug('CREDIT_TYPE : column is not present in the DataFrame')

    def add_features_main(self,main_df):

        '''Extract and Create feature from bureau dataset and append in main dataframe'''
        try:
            # private functions to run so it will extract feature and create a df in loop
            self.feature_extractors   = [
                self._extract_num_credit_currencies,
                self._extract_num_active_credit_d,
                self._extract_days_last_bad_loan,
                self._extract_credit_day_overdue,
                self._extract_days_enddate,
                self._amt_credit_max_overdue,
                self._cnt_credit_prolong,
                self._extract_features_amt_credit,
                self._extract_has_credit_loan
                ]

            self.main_df  = main_df

            for extractor in self.feature_extractors:
                features_df = extractor()
                self.main_df = self.main_df.merge(features_df,on='SK_ID_CURR',how='left')

                # log the method is running
                method_name = extractor.__name__
                logger.info(f"Running {self.__class__.__name__}.{method_name}")
                
                new_cols = features_df.columns.drop('SK_ID_CURR')
                self.main_df[new_cols] = self.main_df[new_cols].fillna(PLACEHOLDER)

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
      

        
    def _extract_number_of_reshedules_tp(self):
        ''' Extract features number of reshedules the client has done in recent time periods
             over all the loans from the 'NUM_INSTALMENT_VERSION' column in the installments payments dataframe.
    
            Features Transformed:
            - NUMBER_OF_RESHEDULES_D: [180, 360, 720, 1080, 2160, 2880] the number of times the loans is resheduled in the recent  months of period

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NUM_INSTALMENT_VERSION' in self.df.columns and 'DAYS_INSTALMENT' in self.df.columns:

            time_frames = [360, 720, 1080, 1440, 1800, 2160]  # 1Y, 2Y, 3Y, 4Y, 5Y, 6Y
            features_df = pd.DataFrame()
            for frame in time_frames:
                filt = (self.df['DAYS_INSTALMENT'] >= -frame) & (self.df['NUM_INSTALMENT_VERSION']!=0)
                filt_df = self.df.loc[filt].copy()
                temp = (filt_df.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].max()-1).to_frame()

                temp = temp.groupby('SK_ID_CURR').sum()
                temp = temp.clip(lower=0)
                feature_df = temp.rename(columns={'NUM_INSTALMENT_VERSION':f'NUMBER_OF_RESHEDULES_{frame}D'})

                 # assign 0 where the num_installment dats are greater than 180 or number of reshedule is 0
            
                if features_df.empty:
                    features_df = feature_df
                else:
                    features_df =  features_df.join(feature_df,how='outer')

            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            features_df = features_df.reindex(all_cust_index,fill_value=0)

            return features_df
           
    
        else:
            logger.debug('NUM_INSTALMENT_VERSION & DAYS_INSTALMENT : column is not present in the DataFrame')



    def _extract_agg_pay_ratio(self):
        ''' extract the pay ratio Avg ,Min,Max PAY_RATIO =  AMT_PAYMENT / AMT_INSTALMENT per customer

            Features Extracted:
            - AVG_PAY_RATIO: average payment ratio per customer
            - MIN_PAY_RATIO: minimum payment ratio per customer
            - MAX_PAY_RATIO: maximum payment ratio per customer

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as index and aggregate PAY_RATIO features
                Missing values filled with the placeholder -99999
        '''

        if 'AMT_PAYMENT' in self.df.columns and 'AMT_INSTALMENT' in self.df.columns:
            
            self.df['PAY_RATIO'] = np.where(self.df['AMT_PAYMENT']!=0,
                     self.df['AMT_PAYMENT'] / self.df['AMT_INSTALMENT'],
                     np.nan
                     )
            
            features_df = self.df.groupby(by='SK_ID_CURR')['PAY_RATIO'].agg(['mean','min','max']).rename(columns={
                "mean":"AVG_PAY_RATIO",
                "min":"MIN_PAY_RATIO",
                "max":"MAX_PAY_RATIO"
            })

            return features_df
            
        else:
            logger.debug('AMT_PAYMENT or DAYS_INSTALMENT : column is not present in the DataFrame')
    
    
    def _extract_early_payments_info(self):
        ''' Extract the early payments ratio and earliest pay flag (3m) from the previous installments dataset
    
            Features Transformed:
            - RATIO_EARLY_PAYMENTS: the ratio of the total payments with payment that is paid before the installment date.
            - RECENT_EARLY_PAYMENT_FLAG_3M : the flag if the person has paid installment early in last 3 months.

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        
        if 'DAYS_ENTRY_PAYMENT' in self.df.columns and 'DAYS_INSTALMENT' in self.df.columns:
            
            # ratio of early payments
            self.df['PAY_DAYS_DIFF'] = self.df['DAYS_ENTRY_PAYMENT']- self.df['DAYS_INSTALMENT']

            self.df['FLAG_PAYMENT_EARLY'] = np.where(self.df['PAY_DAYS_DIFF'] < 0,1,0)


            features_df_1 = self.df.groupby(by='SK_ID_CURR')['FLAG_PAYMENT_EARLY'].mean().to_frame('RATIO_EARLY_PAYMENTS')
            

            # recent early payment 3M
            self.df['RECENT_EARLY_PAYMENT_FLAG_3M'] = np.where(
                    ((self.df['FLAG_PAYMENT_EARLY'] == 1) & (self.df['DAYS_INSTALMENT'] >= -90)),1, 0)
            
            features_df_2 = self.df.groupby(by='SK_ID_CURR')['RECENT_EARLY_PAYMENT_FLAG_3M'].max().to_frame()
            features_df = features_df_1.merge(features_df_2,on='SK_ID_CURR',how='outer')
            return features_df
        else:
            logger.debug('DAYS_ENTRY_PAYMENT or DAYS_INSTALMENT : column is not present in the DataFrame')



    #new feature
    def _extract_worst_dpd_features_installmentsDF(self):
        '''  Create worst DPD (Days Past Due) features for multiple time frames 
            from the installments payments dataset.
    
            Features Extracted:
            - WORST_DPD_ based on time frame: [90, 180, 270, 360, 720, 1080, 2160, 2880] Days

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as index and WORST_DPD_INSTALLMENT_PAYMENTS_XM features
                Missing values filled with the placeholder -99999
        '''

        if 'DAYS_ENTRY_PAYMENT' in self.df.columns and 'DAYS_INSTALMENT' in self.df.columns:
                
            time_frames = [90, 180, 270, 360, 720, 1080, 2160, 2880]
            # empty dataframe to apend the feature into
            
            self.df['PAY_DAYS_DIFF'] = self.df['DAYS_ENTRY_PAYMENT']- self.df['DAYS_INSTALMENT']

            features_df = pd.DataFrame()
            for frame in time_frames:
                filt  = (self.df['PAY_DAYS_DIFF']>= 0) & (self.df['DAYS_INSTALMENT'] > -frame)
                filt_df_frame = self.df.loc[filt]

                feature_dpd = filt_df_frame.groupby('SK_ID_CURR')['PAY_DAYS_DIFF'].max().to_frame(f'WORST_DPD_INSTALLMENT_PAYMENTS_{frame}D')

                if features_df.empty:
                    features_df = feature_dpd
                else:
                    features_df =  features_df.join(feature_dpd,how='outer')
            
            return features_df

    
        else:
            logger.debug('DAYS_ENTRY_PAYMENT or DAYS_INSTALMENT : column is not present in the DataFrame')
        

    def _extract_num_underpaid_installments_D(self):
        ''' Create num_underpaid_installment features over multiple time frame
    
            Features Transformed:
            - NUM_UNDERPAID_INSTALLMENTS_ : Number of underpaid installments based on time frame:[180, 360, 720, 1080, 2160, 2880] D

            Returns:
                DataFrame with SK_ID_CURR as index and WORST_DPD_INSTALLMENT_PAYMENTS_XM features
        '''
        
        if 'AMT_INSTALMENT' in self.df.columns and 'AMT_PAYMENT' in self.df.columns:
            
            time_frames = [180, 360, 720, 1080, 2160, 2880]
            features_df = pd.DataFrame()
            for frame in time_frames:
                filt  = (self.df['DAYS_INSTALMENT'] > -frame)
                filt_df = self.df.loc[filt]
                
                filt_df = filt_df.groupby(['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER'])[['AMT_INSTALMENT','AMT_PAYMENT']].sum()

                filt_df['NUM_UNDERPAID_INSTALLMENTS'] = (filt_df['AMT_INSTALMENT'] > filt_df['AMT_PAYMENT']).astype('int')

                num_underpaid_installments = filt_df.groupby('SK_ID_CURR')['NUM_UNDERPAID_INSTALLMENTS'].sum().to_frame(f'NUM_UNDERPAID_INSTALLMENTS_{frame}D')


                if features_df.empty:
                    features_df = num_underpaid_installments
                else:
                    features_df =  features_df.join(num_underpaid_installments,how='outer').fillna(0).astype(int)
            
            return features_df
        
        else:
            logger.debug('AMT_INSTALMENT or AMT_PAYMENT : column is not present in the DataFrame')

    def _extract_agg_pay_diff(self):
        ''' aggregate min,max,sum,mean  of the PAY_DIFF and extract them
    
            Features Transformed:
            - "SUM_PAY_DIFF": sum of the total pay_diff
            - "MEAN_PAY_DIFF": mean of the total pay_diff
            - "MIN_PAY_DIFF": min of the total pay_diff
            - "MAX_PAY_DIFF": max of the total pay_diff

            Returns:
                DataFrame with SK_ID_CURR as index and aggregated features of the payment difference
        '''
        
        if 'AMT_INSTALMENT' in self.df.columns and 'AMT_PAYMENT' in self.df.columns:
            
            
            filt_df = self.df.groupby(['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER'])[['AMT_INSTALMENT','AMT_PAYMENT']].sum()

            filt_df['PAY_DIFF'] = filt_df['AMT_INSTALMENT'] - filt_df['AMT_PAYMENT']

            features_df =  filt_df.groupby(by='SK_ID_CURR')['PAY_DIFF'].agg(['mean','min','max','sum']).rename(columns= {
                "mean":"MEAN_PAY_DIFF",
                "min":"MIN_PAY_DIFF",
                "max":"MAX_PAY_DIFF",
                "sum":"SUM_PAY_DIFF"
                })


            return features_df
        
        else:
            logger.debug('AMT_INSTALMENT or AMT_PAYMENT : column is not present in the DataFrame')

    
    def add_features_main(self,main_df):
        '''Extract and Create feature from installments payment dataset and append in main application  dataframe'''   
        try:
            self.feature_extractors  = [
                self._extract_number_of_reshedules_tp,
                self._extract_agg_pay_ratio,
                self._extract_early_payments_info,
                self._extract_worst_dpd_features_installmentsDF,
                self._extract_num_underpaid_installments_D,
                self._extract_agg_pay_diff
            ]

            self.main_df = main_df
            for extractor in self.feature_extractors:
                method_name = extractor.__name__

                features_df = extractor()
                logger.info(f"Running {self.__class__.__name__}.{method_name}")

                features_cols = features_df.columns.to_list()

                self.main_df = self.main_df.merge(features_df,on='SK_ID_CURR',how='left')
                self.main_df[features_cols] = self.main_df[features_cols].fillna(PLACEHOLDER)

            logger.info("Aggregated features from the installlments payments dataframe successfully merged into the Previous Application dataframe.")

            return self.main_df
         
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


    def _extract_num_active_loans_XM(self):
        ''' Extract num acitve loans in monthe period not completed
    
            Features Transformed:
            - NUM_ACTIVE_LOANS_XM: count of active loans in the time period [3,6,9,12,24] months

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
            '''
        
        if 'NAME_CONTRACT_STATUS' in self.df.columns  and 'MONTHS_BALANCE' in self.df.columns:
            time_frames = [3,9,12,24]
            features_df = pd.DataFrame()

            for frame in time_frames:

                filt_df = self.df[self.df['MONTHS_BALANCE'] >= -frame].copy()
               
                # vectorized flags
                filt_df['IS_ACTIVE'] = filt_df['NAME_CONTRACT_STATUS'].eq('Active')
                filt_df['IS_COMPLETED'] = filt_df['NAME_CONTRACT_STATUS'].eq('Completed')

                loan_status = filt_df.groupby(['SK_ID_CURR','SK_ID_PREV'])[['IS_ACTIVE','IS_COMPLETED']].max()

                # Active AND not Completed loans
                valid_loans = loan_status[(loan_status['IS_ACTIVE']) & (~loan_status['IS_COMPLETED'])]
                valid_loans = valid_loans.reset_index()
                feature_df = valid_loans.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame(f'NUM_ACTIVE_LOANS_{frame}M')

                if features_df.empty:
                    features_df = feature_df
                else:
                    features_df = features_df.join(feature_df,how='outer')

            all_cust_index = self.df['SK_ID_CURR'].unique()
            features_df = features_df.reindex(all_cust_index,fill_value=0)    
                
            return features_df
        
        else:
            logger.debug('NAME_CONTRACT_STATUS and MONTHS_BALANCE : column is not present in the DataFrame')

    
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

    def _extract_worst_dpd_features_pos_cash(self):
        '''  Create worst DPD (Days Past Due) features for multiple time frames 
            from the POS CASH BALANCE dataset.
    
            Features Extracted:
            - WORST_DPD_POS_CASH_ based on time frame:[3, 6, 9, 12, 24, 36, 72, 96] M

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as index and WORST_DPD_POS_CASH_XM features
                Missing values filled with the placeholder -99999

        '''
        if 'SK_DPD' in self.df.columns and 'MONTHS_BALANCE' in self.df.columns:
            time_frames = [3, 6, 9, 12, 24, 36, 72, 96]
            # empty dataframe to apend the feature into
            features_df = pd.DataFrame()

            for frame in time_frames:
                filt = self.df['MONTHS_BALANCE'] >= -frame
                temp = self.df.loc[filt].copy()
                feature_dpd = temp.groupby(by='SK_ID_CURR')['SK_DPD'].max().to_frame(f'WORST_DPD_POS_CASH_{frame}M')
                
                if features_df.empty:
                    features_df = feature_dpd
                else:
                    features_df =  features_df.join(feature_dpd,how='outer')

            features_df = features_df.sort_index()
            return features_df
        else:
            logger.debug('STATUS : column is not present in the DataFrame')

    def _extract_worst_dpd_def_features_pos_cash(self):
        '''Create severe DPD (Days Past Due with tolerance) features for multiple time frames 
            from the POS_CASH_BALANCE dataset.

            Features Extracted:
            - WORST_DPD_DEF_POS_CASH_XM for time frames [3, 6, 9, 12, 24, 36, 72, 96]

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as index and WORST_DPD_DEF_POS_CASH_XM features
                Missing values filled with the placeholder -99999
            '''
        if 'SK_DPD_DEF' in self.df.columns and 'MONTHS_BALANCE' in self.df.columns:
            time_frames = [3, 6, 9, 12, 24, 36, 72, 96]
            # empty dataframe to apend the feature into
            features_df = pd.DataFrame()

            for frame in time_frames:
                filt = self.df['MONTHS_BALANCE'] >= -frame
                temp = self.df.loc[filt].copy()
                feature_dpd = temp.groupby(by='SK_ID_CURR')['SK_DPD_DEF'].max().to_frame(f'WORST_DPD_DEF_POS_CASH_{frame}M')
                
                if features_df.empty:
                    features_df = feature_dpd
                else:
                    features_df =  features_df.join(feature_dpd,how='outer')

            features_df = features_df.sort_index()
            return features_df
        else:
            logger.debug('STATUS and SK_DPD_DEF: column is not present in the DataFrame')
                  

    def add_features_main(self,main_df):

        '''Extract and Create feature Pos Cash Balance and append in Main Dataframe'''   
        try:
            self.feature_extractors  = [
                self._extract_has_risky_contract_status,
                self._extract_num_active_loans_XM,
                self._extract_cnt_installment_future,
                self._extract_worst_dpd_features_pos_cash,
                self._extract_worst_dpd_def_features_pos_cash
            ]

           
            self.main_df = main_df
            for extractor in self.feature_extractors:
                features_df = extractor()
                features_cols = features_df.columns.to_list()
                # log the method is running
                method_name = extractor.__name__
                logger.info(f"Running {self.__class__.__name__}.{method_name}")
                self.main_df = self.main_df.merge(features_df,on='SK_ID_CURR',how='left')
                self.main_df[features_cols] = self.main_df[features_cols].fillna(PLACEHOLDER)

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

    def _safe_join(self,base,new)-> pd.DataFrame:
        ''' helper function'''
        if base.empty:
            return new
        else:
            return base.join(new, how='outer')
        
    def _extract_flag_has_credit_history(self):

        ''' flag the client if the client have any credit card history from the previous application dataset

            Features Transformed:
            - FLAG_HAS_CREDIT_CARD_HISTORY : flag 1 if person own the credit card  history else 0 

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'NAME_PORTFOLIO'}
        if required_cols.issubset(self.df.columns):

            self.df['FLAG_HAS_CREDIT_CARD_HISTORY'] = (self.df['NAME_PORTFOLIO'] =='Credit').astype(int)
            feature_df = self.df.groupby('SK_ID_CURR')['FLAG_HAS_CREDIT_CARD_HISTORY'].max().to_frame()

            return feature_df

        else:
            logger.debug(" NAME_PORTFOLIO column are not present in the DataFrame")   
    #new

    def _extract_annuity_features(self):
        '''
            Extract aggregated annuity-related features from the previous application dataset.

            portfolio_types = ['POS', 'Cash', 'Cards']

            Features Transformed:
            - AVG_AMT_ANNUITY_POS_{portfolio}: Average annuity amount (AMT_ANNUITY) for portfolio loans per client.       
            - AVG_AMT_CREDIT_ANNUITY_RATIO_{portfolio}:Average ratio of sum(AMT_CREDIT) / sum(AMT_ANNUITY) for portfolio loans
    
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and aggregated features
        '''
        if 'AMT_ANNUITY' in self.df.columns :

            category_types = ['POS', 'Cash', 'Cards']
            features_df = pd.DataFrame()
            
            # avg annuity ( installments amount) for pos,cash,cards categories
            for category in category_types:

                filt = (self.df['NAME_PORTFOLIO'] == category) & (self.df['AMT_ANNUITY'] > 0)
                filt_df = self.df.loc[filt].copy()

                agg_df = filt_df.groupby(by='SK_ID_CURR').agg(
                    AVG_AMT_ANNUITY = ('AMT_ANNUITY', 'mean'),
                    SUM_AMT_CREDIT=('AMT_CREDIT', 'sum'),
                    SUM_AMT_ANNUITY=('AMT_ANNUITY', 'sum')
                    ).reset_index()

            
                agg_df[f'AMT_CREDIT_TO_ANNUITY_RATIO'] = agg_df['SUM_AMT_CREDIT'] / agg_df['SUM_AMT_ANNUITY']

                agg_df.rename(columns={
                    'AMT_CREDIT_TO_ANNUITY_RATIO':f'AMT_CREDIT_TO_ANNUITY_RATIO_{category.upper()}',
                    'AVG_AMT_ANNUITY':f'AVG_AMT_ANNUITY_{category.upper()}'
                },inplace=True)
                
                feature_df = agg_df[['SK_ID_CURR',f'AMT_CREDIT_TO_ANNUITY_RATIO_{category.upper()}',f'AVG_AMT_ANNUITY_{category.upper()}']]

                if features_df.empty:
                    features_df = feature_df
                else:
                    features_df =  features_df.merge(feature_df,on='SK_ID_CURR',how='outer')  

            return features_df
        
        else:
            logger.debug('AMT_ANNUITY : column is not present in the DataFrame')

    def _extract_loan_counts_by_category_M(self):

        ''' Extract number of loans per category the customer had in last [1,2,3,4] Year features
            in the previous application dataset.

            Features Transformed:
            - NUM_LOANS_{CATEGORY}_{PERIOD}: num loans per category and with  time frames

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NAME_PORTFOLIO' in self.df.columns and 'DAYS_DECISION' in self.df.columns:
            category_types = ['POS', 'Cash', 'Cards']
            time_frames = [360, 720, 1080, 1440, 1800]  # 1Y, 2Y, 3Y, 4Y, 5Y

            features_df = pd.DataFrame()
            for category in category_types:
                for frame in time_frames:
                    filt = ((self.df['NAME_PORTFOLIO'] == category) & (self.df['DAYS_DECISION'] > - frame))
                    filt_df = self.df.loc[filt].copy()

                    feature_df = filt_df.groupby(by='SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame(f'NUM_LOANS_{category}_{frame}D')
                    
                    features_df = self._safe_join(features_df,feature_df)

            return features_df.fillna(0)
        
        else:
            logger.debug('DAYS_DECISION and NAME_PORTFOLIO : column are not present in the DataFrame')

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
        ''' Extract ratio of the credit / application for different categories such as pos,cash,credit
            from previous application dataset

            Features Transformed:
            - AVG_CREDIT_APPLICATION_RATIO:  AMT_CREDIT / AMT_APPLICATION and the average of this value per customer

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'AMT_CREDIT' in self.df.columns and 'AMT_APPLICATION' in self.df.columns  :
            category_types = ['POS', 'Cash', 'Cards']
            features_df = pd.DataFrame()

            for category in category_types:
                filt = (self.df['NAME_PORTFOLIO'] == category) & (self.df['AMT_APPLICATION']>0)
                filt_df = self.df.loc[filt].copy()

                agg_df = filt_df.groupby(by='SK_ID_CURR').agg(
                    SUM_AMT_CREDIT = ('AMT_CREDIT','sum'),
                    SUM_AMT_APPLICATION= ('AMT_APPLICATION','sum')
                )

                agg_df[f'CREDIT_APPLICATION_RATIO_{category}'] = agg_df['SUM_AMT_CREDIT'] / agg_df['SUM_AMT_APPLICATION']
                feature_df = agg_df[[f'CREDIT_APPLICATION_RATIO_{category}']]

                features_df = self._safe_join(features_df,feature_df)
            return features_df
        else:
            logger.debug('AMT_CREDIT and  AMT_APPLICATION : column are not present in the DataFrame')

   

    def _extract_avg_down_payment_rate(self):
        ''' Extract features from the 'RATE_DOWN_PAYMENT' in the previous application dataset.
            downpayment is done for the POS category only

            Features Transformed:
            - AVG_DOWN_PAYMENT_RATE:  avg downpayment rate for the per client 
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'RATE_DOWN_PAYMENT' in self.df.columns and 'NAME_PORTFOLIO' in self.df.columns:

            filt_df = self.df[self.df['NAME_PORTFOLIO'] =='POS']

            feature_df =  filt_df.groupby(by='SK_ID_CURR')['RATE_DOWN_PAYMENT'].mean().to_frame('AVG_DOWN_PAYMENT_RATE')

            
            return feature_df
        
        else:
            logger.debug('RATE_DOWN_PAYMENT and NAME_PORTFOLIO : column are not present in the DataFrame')
            

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


    def _extract_approved_refused_loan_ratios(self):
        ''' Extract  
                approved / total loan ratio ,
                refused / total loan ratio 
                from the previous application dataset.

            Features Transformed:
            - RATIO_REFUSED_LOANS: ratio of the refused loans.  refused / approved + refused
            - RATIO_APPROVED_LOANS: ratio of the refused loans.  approved / approved + refused

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'NAME_CONTRACT_STATUS' in self.df.columns:
            temp = pd.crosstab(self.df['SK_ID_CURR'],self.df['NAME_CONTRACT_STATUS'],dropna=False)
            temp['TOTAL_LOANS']=temp[['Approved','Refused']].sum(axis=1)
        
            temp['RATIO_REFUSED_LOANS'] = temp['Refused'] / temp['TOTAL_LOANS']
            temp['RATIO_APPROVED_LOANS'] = temp['Approved'] / temp['TOTAL_LOANS']

            features_df =  temp[['RATIO_REFUSED_LOANS','RATIO_APPROVED_LOANS']].copy()

            return features_df
        else:
            logger.debug('NAME_CONTRACT_STATUS  : column are not present in the DataFrame')


    def _extract_num_refused_loans_d(self):

        ''' Extract features from the 'DAYS_DECISION' and  NAME_CONTRACT_STATUS in the previous application dataset.

            Features Transformed:
            - LOANS_REFUSED_RECENT_D: num of refused loans in the last days_window ex: 60 d 180 d 

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        
        if 'DAYS_DECISION' in self.df.columns and 'NAME_CONTRACT_STATUS' in self.df.columns :
            time_frames = [180,360, 720, 1080, 1440] 
            features_df = pd.DataFrame()

            for frame in time_frames:
                    
                filt = ((self.df['DAYS_DECISION'] > -frame) & (self.df['NAME_CONTRACT_STATUS'] == 'Refused'))

                temp = self.df.loc[filt].copy()

                feature_df = temp.groupby(by='SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame(f'LOANS_REFUSED_RECENT_{frame}D')
                
                features_df = self._safe_join(features_df,feature_df)
                
            return features_df.fillna(0)
        
        else:
            logger.debug('NAME_CONTRACT_STATUS and DAYS_DECISION : column are not present in the DataFrame')


    def _extract_days_since_last_loan(self):
        ''' Extract the recency of the client's loan applications from the previous application dataset.

            Features Transformed:
            - DAYS_SINCE_LAST_LOAN_APPLY :Number of days since the client last applied for any loan.
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'DAYS_DECISION' in self.df.columns:
            feature_df = self.df.groupby('SK_ID_CURR')['DAYS_DECISION'].max().to_frame('DAYS_SINCE_LAST_LOAN_APPLY')

            feature_df['DAYS_SINCE_LAST_LOAN_APPLY'] = -feature_df['DAYS_SINCE_LAST_LOAN_APPLY']

            return feature_df
        else:
            logger.debug('DAYS_DECISION column not present in the DataFrame')

    def _extract_num_hc_reject_loans_d(self):

        ''' extract the number of loans rejected due to HC reject reason per client  in time frames

            Features Transformed:
            - NUM_HC_REJECT_REASON_XD: Flag if the client has the loan  rejected loan due to the HC reject reason in last 2 year
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'CODE_REJECT_REASON' in self.df.columns and 'DAYS_DECISION' in self.df.columns:
            time_frames =[90, 180, 270, 360, 720]
            features_df = pd.DataFrame({'SK_ID_CURR': self.df['SK_ID_CURR'].unique()})

            self.df['FLAG_HC_REJECT_REASON'] = (self.df['CODE_REJECT_REASON'] =='HC').astype(int)

            for frame in time_frames:    
                filt = (self.df['DAYS_DECISION'] > -frame)
                temp = (self.df.loc[filt]).copy()

                feature_df = temp.groupby(by='SK_ID_CURR')['FLAG_HC_REJECT_REASON'].sum().to_frame(f'NUM_HC_REJECT_REASON_{frame}D').reset_index()

                features_df = features_df.merge(feature_df, on='SK_ID_CURR', how='left')
        
            return features_df.fillna(0)
        
        else:
            logger.debug('CODE_REJECT_REASON and DAYS_DECISION : column are not present in the DataFrame')
    
    def _extract_ratio_hc_reject_loans(self):
        '''  Extract the ratio of loans rejected due to HC reject reason per client.

            Features Transformed:
            - RATIO_HC_REFUSED_LOANS: ratio of HC refused loans / total refused loans
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''            

        if 'CODE_REJECT_REASON' in self.df.columns:
            filt_df = self.df.copy()
            filt_df['NUM_HC_REJECT_REASON'] = (filt_df['CODE_REJECT_REASON'] =='HC').astype(int)
            filt_df['NUM_REFUSED'] = (filt_df['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
            
            feature_df = filt_df.groupby(by='SK_ID_CURR').agg(
                    NUM_HC_REJECT_REASON_LOANS =('NUM_HC_REJECT_REASON','sum'),
                    NUM_REFUSED_LOANS =('NUM_REFUSED','sum'),
                ).reset_index()
            feature_df['RATIO_HC_REFUSED_LOANS'] =feature_df['NUM_HC_REJECT_REASON_LOANS'] / feature_df['NUM_REFUSED_LOANS']
            feature_df = feature_df[['SK_ID_CURR','RATIO_HC_REFUSED_LOANS']]

            return feature_df
        else:
            logger.debug('CODE_REJECT_REASON column are not present in the DataFrame')

    
    def _extract_num_sco_scofr_reject_loans(self):

        ''' Extract the count of the rejected loans  when the reject reason was the sco,scofr
            from the NUM_SCO_SCOFR_REJECT_REASON feature

            Features Transformed:
            - NUM_SCO_SCOFR_REJECT_REASON: total num of loan rejected for the SCO or SCOFR reason per client. 
            
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'CODE_REJECT_REASON' in self.df.columns :

            self.df['NUM_SCO_SCOFR_REJECT_REASON'] = np.where(self.df['CODE_REJECT_REASON'].isin(['SCO','SCOFR']),
                                       1,
                                       0)
           
            feature_df =self.df.groupby(by='SK_ID_CURR')['NUM_SCO_SCOFR_REJECT_REASON'].sum().to_frame().reset_index()

            return feature_df.fillna(0)
        
        else:
            logger.debug('CODE_REJECT_REASON : column are not present in the DataFrame')
    
    def _extract_num_limit_reject_loans(self):

        ''' Extract the count of the rejected loans  when the reject reason was the limit

            Features Transformed:
            - NUM_LIMIT_REJECT_REASON: total num of loan rejected for the LIMIT reason per client. 
            
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'CODE_REJECT_REASON' in self.df.columns :

            self.df['NUM_LIMIT_REJECT_REASON'] = (self.df['CODE_REJECT_REASON'] == 'LIMIT').astype(int)
           
            feature_df =self.df.groupby(by='SK_ID_CURR')['NUM_LIMIT_REJECT_REASON'].sum().to_frame().reset_index()
            return feature_df.fillna(0)
        else:
            logger.debug('CODE_REJECT_REASON : column are not present in the DataFrame')

    def _extract_unknown_reject_reason_cnt(self):
        ''' Count the number of loans per client where the reject reason is missing or unknown (NaN, XNA, XAP). 
            
            Features Transformed:
            - UNKNOWN_REJECT_REASON_CNT:  the count of the loans where the reject is unknows i.e NaN,XNA,XPA
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        
        '''
        if 'CODE_REJECT_REASON' in self.df.columns:
            self.df['UNKNOWN_REJECT_REASON_CNT'] = self.df['CODE_REJECT_REASON'].isna() | self.df['CODE_REJECT_REASON'].isin(['XNA', 'XAP'])
            self.df['UNKNOWN_REJECT_REASON_CNT'] = self.df['UNKNOWN_REJECT_REASON_CNT'].astype(int)
            
            feature_df = self.df.groupby('SK_ID_CURR')['UNKNOWN_REJECT_REASON_CNT'].sum().to_frame().reset_index()
            return feature_df.fillna(0)
        else:
            logger.debug('CODE_REJECT_REASON column not present in the DataFrame')


    def _extract_ratio_repeater(self):
        ''' Extract overall repeater application ratio per client.

        Features Transformed:
        - RATIO_REPEATER: ratio of applications where client was a Repeater 
                          over total applications (full history)

        Returns:
            features_df (pd.DataFrame): DataFrame with SK_ID_CURR index
        '''

        if 'NAME_CLIENT_TYPE' in self.df.columns:

            df = self.df.copy()
            df['FLAG_REPEATER'] = (df['NAME_CLIENT_TYPE'] == 'Repeater').astype(int)

            feature_df = (
                df.groupby('SK_ID_CURR')['FLAG_REPEATER']
                .mean()
                .to_frame('RATIO_REPEATER')
            ).reset_index()

            return feature_df

        else:
            logger.debug('NAME_CLIENT_TYPE column not present in the DataFrame')


    def _extract_flag_new_client_1y(self):
        ''' Flag if the client applied as a New client in last 1 year.

            Features Transformed:
            - FLAG_NEW_CLIENT_1Y: 1 if client was new in last 365 days else 0

            Returns:
                features_df (pd.DataFrame): DataFrame with SK_ID_CURR index
        '''

        if {'NAME_CLIENT_TYPE', 'DAYS_DECISION'}.issubset(self.df.columns):

            df = self.df.copy()

            df['FLAG_NEW_CLIENT_1Y'] = (
                (df['DAYS_DECISION'] > -365) &
                (df['NAME_CLIENT_TYPE'] == 'New')
            ).astype(int)

            feature_df = (
                df.groupby('SK_ID_CURR')['FLAG_NEW_CLIENT_1Y']
                .max()
                .to_frame()
            ).reset_index()

            return feature_df

        else:
            logger.debug('Required columns missing: NAME_CLIENT_TYPE or DAYS_DECISION')

    def _extract_ratio_high_risk_channel(self):
        '''
        Extract the ratio of applications made via high-risk channels per client.

        High-risk channels:
        - AP+ (Cash loan)
        - Contact center

        Features Transformed:
        - RATIO_HIGH_RISK_CHANNEL: Number of applications via high-risk channels / Total number of applications

        Returns:
            features_df (pd.DataFrame): DataFrame with SK_ID_CURR index
        '''

        if 'CHANNEL_TYPE' in self.df.columns:

            df = self.df.copy()

            df['FLAG_HIGH_RISK_CHANNEL'] = df['CHANNEL_TYPE'].isin(
                ['AP+ (Cash loan)', 'Contact center']
            ).astype(int)

            feature_df = df.groupby('SK_ID_CURR').agg(
                NUM_HIGH_RISK_CHANNEL=('FLAG_HIGH_RISK_CHANNEL', 'sum'),
                TOTAL_APPLICATIONS=('FLAG_HIGH_RISK_CHANNEL', 'count')
            ).reset_index()

            feature_df['RATIO_HIGH_RISK_CHANNEL'] = (
                feature_df['NUM_HIGH_RISK_CHANNEL'] /
                feature_df['TOTAL_APPLICATIONS']
            )

            return feature_df[['SK_ID_CURR','RATIO_HIGH_RISK_CHANNEL']]

        else:
            logger.debug('Required columns  oHANNEL_TYPE not present in DataFrame')

    def _extract_ratio_insured_loans(self):

        ''' Extract features from the NFLAG_INSURED_ON_APPROVAL  in the previous application dataset.

            Features Transformed:
            - RATIO_INSURED_LOANS: ratio of the insured loans of the client

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NFLAG_INSURED_ON_APPROVAL' in self.df.columns:
            
            feature_df = self.df.groupby('SK_ID_CURR')['NFLAG_INSURED_ON_APPROVAL'].mean().to_frame('RATIO_INSURED_LOANS').reset_index()

            return feature_df
        
        else:
            logger.debug('NFLAG_INSURED_ON_APPROVAL  : column are not present in the DataFrame')

    def _extract_suite_behavior_features(self):
        '''
        Extract client behavior features based on NAME_TYPE_SUITE
        (who accompanied the client during loan applications).

        Features Transformed:
        - FLAG_ALWAYS_UNACCOMPANIED: 1 if client always applied unaccompanied
        - FLAG_NEVER_UNACCOMPANIED: 1 if client never applied unaccompanied
        - RATIO_UNACCOMPANIED: ratio of applications where client was unaccompanied
        - N_UNIQUE_SUITES: number of unique accompaniment types per client

        Returns:
            features_df (pd.DataFrame): DataFrame with SK_ID_CURR index
        '''

        required_cols = {'SK_ID_CURR', 'NAME_TYPE_SUITE'}

        if required_cols.issubset(self.df.columns):
        
            filt = self.df['NAME_TYPE_SUITE'].isin(['XNA','XAP'])
            df = self.df.loc[~filt].copy()
            # Flag unaccompanied applications
            df['FLAG_UNACCOMPANIED'] = (df['NAME_TYPE_SUITE'] == 'Unaccompanied').astype(int)
            
            agg_df = df.groupby('SK_ID_CURR').agg(
                FLAG_ALWAYS_UNACCOMPANIED=('FLAG_UNACCOMPANIED', 'min'),
                FLAG_SUM = ('FLAG_UNACCOMPANIED','sum'),
                RATIO_UNACCOMPANIED=('FLAG_UNACCOMPANIED', 'mean'),
                N_UNIQUE_SUITES=('NAME_TYPE_SUITE', 'nunique')
                ).reset_index()

            # always alone the if min == 1 then the client was always alone while applicatoin
            agg_df['FLAG_ALWAYS_UNACCOMPANIED'] = (agg_df['FLAG_ALWAYS_UNACCOMPANIED']==1).astype(int)
            # if the client dont have unaccompanied flag then the value for flag sum will be 0
            agg_df['FLAG_NEVER_UNACCOMPANIED'] = (agg_df['FLAG_SUM'] == 0).astype(int)

            features_df = agg_df[['SK_ID_CURR','FLAG_ALWAYS_UNACCOMPANIED','FLAG_NEVER_UNACCOMPANIED','RATIO_UNACCOMPANIED','N_UNIQUE_SUITES']]

            return features_df
        

        else:
            logger.debug('NAME_TYPE_SUITE column are not present in the DataFrame')


    def _extract_ratio_highrisk_yield_loans_d(self):

        ''' Extract ratio of the high_risk loans based on time frames 1,2,3,4 years

            Features Transformed:
            - RATIO_HIGH_RISK_YIELD_LOANS_XD: ratio high_risk loans / total loans time frames

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NAME_YIELD_GROUP' in self.df.columns:
            time_frames = [360, 720, 1080] 
            features_df = pd.DataFrame()

            self.df['HIGH_RISK_YIELD_LOANS'] = (self.df['NAME_YIELD_GROUP']=='high').astype(int)

            for frame in time_frames:
                filt = self.df['DAYS_DECISION']> - frame
                filt_df = self.df.loc[filt].copy()
                feature_df = filt_df.groupby('SK_ID_CURR')['HIGH_RISK_YIELD_LOANS'].mean().to_frame(f'RATIO_HIGH_RISK_YIELD_LOANS_{frame}D')

                features_df = self._safe_join(features_df,feature_df)

                
            return features_df
        
        else:
            logger.debug('NAME_YIELD_GROUP  : column are not present in the DataFrame')

    def _extract_avg_risk_weight(self):
        '''
        Calculate average risk weight per client using NAME_YIELD_GROUP.

        Features Transformed:
        - AVG_RISK_WEIGHT: average risk per client by encoding the name yield group into ordinal numbers

        Returns:
            features_df(pd.DataFrame): DataFrame with SK_ID_CURR index
                                    and AVG_RISK_WEIGHT feature
        '''
        if 'NAME_YIELD_GROUP'  in self.df.columns:

            risk_map = {
                'high': 4,
                'middle': 3,
                'low_normal': 2,
                'low_action': 1,
                'XNA': 0
            }
            
            time_frames_days = [180, 360, 720, 1080]  
            features_df = pd.DataFrame()
            df = self.df.copy()
            df['RISK_WEIGHT'] = df['NAME_YIELD_GROUP'].map(risk_map)

            for frame in time_frames_days:
                filt_df = df[df['DAYS_DECISION'] > -frame]
                # Group by client and calculate average
                agg_df = (
                    filt_df.groupby('SK_ID_CURR')['RISK_WEIGHT']
                    .mean()
                    .to_frame(f'AVG_RISK_WEIGHT_{frame}D')
                )
                if features_df.empty:
                    features_df = agg_df
                else:
                    features_df = features_df.join(agg_df,how='outer')

            return features_df 
        
        else:
            logger.debug('NAME_YIELD_GROUP column is not present')

    def _extract_avg_max_loan_delay(self):
        '''
        Extract mean and max loan repayment delays per client and per loan category.

        Features Transformed:
            - MEAN_LOAN_REPAYMENT_DIFF_<CATEGORY>
            - MAX_LOAN_REPAYMENT_DIFF_<CATEGORY>

        Returns:
            features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'DAYS_TERMINATION' in self.df.columns and  'DAYS_LAST_DUE' in self.df.columns:

            self.df['DAYS_TERMINATION'] = self.df['DAYS_TERMINATION'].replace({365243.0:np.nan})
            self.df['DAYS_LAST_DUE'] = self.df['DAYS_LAST_DUE'].replace({365243.0:np.nan})
            categories = ['POS', 'Cash', 'Credit'] 
            features_df = pd.DataFrame()
            self.df['LOAN_REPAYMENT_DIFF'] = self.df['DAYS_TERMINATION'] - self.df['DAYS_LAST_DUE']

            for category in categories:
                filt = (self.df['NAME_PORTFOLIO'] == category)
                filt_df = self.df.loc[filt].copy()
        

                agg_df = filt_df.groupby('SK_ID_CURR')['LOAN_REPAYMENT_DIFF'].agg(['mean','max']).rename(columns={
                    "mean":f"MEAN_LOAN_REPAYMENT_DIFF_{category}",
                    "max":f"MAX_LOAN_REPAYMENT_DIFF_{category}"
                })

                if features_df.empty:
                    features_df = agg_df
                else:
                    features_df = features_df.join(agg_df,how='outer')

            return features_df
        
        else:
            logger.debug('DAYS_TERMINATION and  DAYS_LAST_DUE : column are not present in the DataFrame')

     
 
    def add_features_main(self,main_df):

        '''Extract and Create feature from Previous Application dataset and append in Main Dataframe'''   
        try:
            self.feature_extractors  = [
                self._extract_flag_has_credit_history,
                self._extract_annuity_features,
                self._extract_loan_counts_by_category_M,
                self._extract_avg_credit_client,
                self._extract_avg_credit_application_ratio,
                self._extract_avg_down_payment_rate,
                self._extract_avg_goods_price,
                self._extract_mean_privileged_interest_rate,
                self._extract_approved_refused_loan_ratios,
                self._extract_num_refused_loans_d,
                self._extract_days_since_last_loan,
                self._extract_num_hc_reject_loans_d,
                self._extract_ratio_hc_reject_loans,
                self._extract_num_sco_scofr_reject_loans,
                self._extract_num_limit_reject_loans,
                self._extract_unknown_reject_reason_cnt,
                self._extract_ratio_repeater,
                self._extract_flag_new_client_1y,
                self._extract_ratio_high_risk_channel,
                self._extract_ratio_insured_loans,
                self._extract_suite_behavior_features,
                self._extract_ratio_highrisk_yield_loans_d,
                self._extract_avg_risk_weight,
                self._extract_avg_max_loan_delay,
            ]
            
            self.main_df = main_df
            for extractor in self.feature_extractors:
                method_name = extractor.__name__
                logger.info(f"Running {self.__class__.__name__}.{method_name}")

                features_df = extractor()
                features_cols = features_df.columns.to_list()

                # log the method is running
               
                self.main_df = self.main_df.merge(features_df,on='SK_ID_CURR',how='left')
                self.main_df[features_cols] = self.main_df[features_cols].fillna(PLACEHOLDER)

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
        self.time_frames = [1,3,6,9,12]
     
    @property 
    def monthly_credit_agg(self):
        '''
        Prepare reusable monthly-level credit card aggregates.
        
        Aggregation level:
        - SK_ID_CURR
        - MONTHS_BALANCE
        
        Aggregated columns:
        - AMT_DRAWINGS_CURRENT_TOTAL: Total amount drawn on the credit card during the month
        - AMT_CREDIT_LIMIT_ACTUAL_TOTAL:Total available credit limit across all active credit lines
        - AMT_DRAWINGS_ATM_CURRENT_TOTAL: Total cash withdrawn via ATM
        - AMT_DRAWINGS_POS_CURRENT_TOTAL:Total amount spent via POS transactions
        - AMT_PAYMENT_CURRENT_TOTAL:Total payments made by the customer in the month
        - AMT_INST_MIN_REGULARITY_TOTAL:Minimum required installment payment
        - AMT_BALANCE_TOTAL: Outstanding credit card balance at month-end
        - AMT_RECEIVABLE_PRINCIPAL_TOTAL:Principal amount still receivable

        '''
        
        if not hasattr(self, "_monthly_credit_agg"):
            self._monthly_credit_agg = (
            self.df
            .groupby(['SK_ID_CURR', 'MONTHS_BALANCE'], as_index=False)
            .agg(
                AMT_DRAWINGS_CURRENT_TOTAL=('AMT_DRAWINGS_CURRENT', 'sum'),
                AMT_DRAWINGS_ATM_CURRENT_TOTAL=('AMT_DRAWINGS_ATM_CURRENT', 'sum'),
                AMT_DRAWINGS_POS_CURRENT_TOTAL=('AMT_DRAWINGS_POS_CURRENT', 'sum'),
                AMT_PAYMENT_CURRENT_TOTAL=('AMT_PAYMENT_CURRENT', 'sum'),
                AMT_INST_MIN_REGULARITY_TOTAL=('AMT_INST_MIN_REGULARITY', 'sum'),
                AMT_BALANCE_TOTAL=('AMT_BALANCE', 'sum'),
                AMT_RECEIVABLE_PRINCIPAL_TOTAL=('AMT_RECEIVABLE_PRINCIPAL', 'sum'),
                AMT_CREDIT_LIMIT_ACTUAL_TOTAL=('AMT_CREDIT_LIMIT_ACTUAL', 'sum'),))
        return self._monthly_credit_agg

    def _safe_join(self,base,new)-> pd.DataFrame:
        ''' helper function'''
        if base.empty:
            return new
        else:
            return base.join(new, how='outer')

    def _extract_credit_utilization_features(self):

        ''' Extract maximum weighted credit utilization over multiple time frames.
            time frames :  [3,6,9,12] months
            and trend of credit utilization 3M -> 12M

            WEIGHTED_UTILIZATION = sum(AMT_BALANCE) / sum(AMT_CREDIT_LIMIT_ACTUAL)

            Features Transformed:
            - MAX_WEIGHTED_CREDIT_UTIL_{XM} : max Average ratio utilization per customer across their active lines on time frames
            - CREDIT_UTIL_TREND_3M_12M: Difference between 3-month and 12-month mean utilization,
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE'}

        if required_cols.issubset(self.df.columns):
            # Produces an average utilization per customer across their active lines.
            features_df = pd.DataFrame()

            monthly_df  = self.monthly_credit_agg
            

            for frame in self.time_frames:

                monthly_frame = monthly_df[monthly_df['MONTHS_BALANCE'] >= -frame].copy()

            
                monthly_frame['UTILIZATION_RATIO'] = np.where(
                            monthly_frame['AMT_CREDIT_LIMIT_ACTUAL_TOTAL'] > 0,
                            monthly_frame['AMT_BALANCE_TOTAL'] / monthly_frame['AMT_CREDIT_LIMIT_ACTUAL_TOTAL'],
                            np.nan
                        )           
                
                if frame in [3,12]:
                    feature_df = monthly_frame.groupby('SK_ID_CURR')['UTILIZATION_RATIO'].mean().to_frame(f'WEIGHTED_AVG_CREDIT_UTILIZATION_{frame}M')
                    
                    #safe join helper function
                    features_df = self._safe_join(features_df, feature_df)

                feature_df = monthly_frame.groupby('SK_ID_CURR')['UTILIZATION_RATIO'].max().to_frame(f'MAX_WEIGHTED_CREDIT_UTIL_{frame}M')
                
                features_df = self._safe_join(features_df, feature_df)

            features_df['CREDIT_UTILIZATION_TREND_3M_12M'] = features_df['WEIGHTED_AVG_CREDIT_UTILIZATION_3M'] - features_df['WEIGHTED_AVG_CREDIT_UTILIZATION_12M']
            
            return features_df
        
        else:
            logger.debug("'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE' column are not present in the DataFrame")

    def _extract_credit_usage_features(self):

        ''' Extract credit usage features based on AMT_DRAWINGS_CURRENT and AMT_CREDIT_LIMIT_ACTUAL
            from the Credit Balance dataset.
            Time frames considered: [3, 6, 12, 24]

            Weighted monthly usage ratio is computed as:
            CREDIT_USAGE_RATIO = sum(AMT_DRAWINGS_CURRENT) / sum(AMT_CREDIT_LIMIT_ACTUAL)

            Features Transformed:
            - MAX_CREDIT_USAGE_RATIO_XM:  Maximum monthly weighted  usage ratio over X months.
            - CREDIT_USAGE_TREND_3M_12M : Difference between 3-month and 12-month average usage ratios
                (indicates increasing or decreasing spending behavior).

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT', 'AMT_CREDIT_LIMIT_ACTUAL'}
        if required_cols.issubset(self.df.columns):
            features_df = pd.DataFrame()

            monthly_df  = self.monthly_credit_agg

            for frame in self.time_frames:

                filt  = monthly_df['MONTHS_BALANCE'] >= -frame
                                
                monthly_frame = monthly_df.loc[filt].copy()
            
                monthly_frame['CREDIT_USAGE_RATIO_PER_MONTH'] = np.where(
                            monthly_frame['AMT_CREDIT_LIMIT_ACTUAL_TOTAL'] > 0,
                            monthly_frame['AMT_DRAWINGS_CURRENT_TOTAL'] / monthly_frame['AMT_CREDIT_LIMIT_ACTUAL_TOTAL'],
                            np.nan
                        )  
                if frame in [3,12]:
                    feature_df = monthly_frame.groupby('SK_ID_CURR')['CREDIT_USAGE_RATIO_PER_MONTH'].mean().to_frame(f'AVG_CREDIT_DRAWING_RATIO_{frame}M')
                    features_df = self._safe_join(features_df, feature_df)

                
                feature_df = monthly_frame.groupby('SK_ID_CURR')['CREDIT_USAGE_RATIO_PER_MONTH'].max().to_frame(f'MAX_CREDIT_DRAWING_RATIO_{frame}M')
                features_df = self._safe_join(features_df, feature_df)


            features_df['CREDIT_DRAWING_TREND_3M_12M'] = features_df['AVG_CREDIT_DRAWING_RATIO_3M'] - features_df['AVG_CREDIT_DRAWING_RATIO_12M']

            
            return features_df
        else:
            logger.debug("'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL',AMT_DRAWINGS_CURRENT  column are not present in the DataFrame")

     
    def _extract_atm_cash_usage_features(self):

        '''  Extract ATM cash utilization features from the Credit Card Balance dataset.

            Features Transformed:
            - MAX_ATM_CASH_UTILIZATION_RATIO_XM:  Maximum ATM cash monthly usage ratio over X months.
            - AVG_ATM_CASH_UTILIZATION_RATIO_3M_12M: Difference between 3-month and 12-month average ATM cash utilization.
                     Positive value indicates increasing reliance on ATM cash.

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_CREDIT_LIMIT_ACTUAL'}

        if required_cols.issubset(self.df.columns):
            features_df = pd.DataFrame()

            monthly_df  = self.monthly_credit_agg

            for frame in self.time_frames:

                frame_df = monthly_df[monthly_df['MONTHS_BALANCE'] >= -frame].copy()


                frame_df['ATM_CASH_UTILIZATION_RATIO'] = np.where(
                            frame_df['AMT_CREDIT_LIMIT_ACTUAL_TOTAL'] > 0,
                            frame_df['AMT_DRAWINGS_ATM_CURRENT_TOTAL'] / frame_df['AMT_CREDIT_LIMIT_ACTUAL_TOTAL'],
                            np.nan
                        )  
            
                if frame in [3,12]:
                    feature_df = frame_df.groupby('SK_ID_CURR')['ATM_CASH_UTILIZATION_RATIO'].mean().to_frame(f'AVG_ATM_CASH_UTILIZATION_RATIO_{frame}M')
                    features_df = self._safe_join(features_df, feature_df)

                
                feature_df = frame_df.groupby('SK_ID_CURR')['ATM_CASH_UTILIZATION_RATIO'].max().to_frame(f'MAX_ATM_CASH_UTILIZATION_RATIO_{frame}M')

                features_df = self._safe_join(features_df, feature_df)


            features_df['AVG_ATM_CASH_UTILIZATION_RATIO_3M_12M'] = features_df['AVG_ATM_CASH_UTILIZATION_RATIO_3M'] - features_df['AVG_ATM_CASH_UTILIZATION_RATIO_12M']

            return features_df
        
        else:
            logger.debug("'MONTHS_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT' AMT_CREDIT_LIMIT_ACTUAL ,  column are not present in the DataFrame")


    def _extract_avg_atm_withdrawal_frequency(self):
    
        '''  Extract the avg frequency of the atm withdrawl of  client in time frames features from the Credit Card Balance dataset.

            time_frames = [3,6,12,24]

            Features Transformed:
            - AVG_ATM_WITHDRAWAL_FREQ_{XM}:  Count of times the atm is used in last x monhts 

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'CNT_DRAWINGS_ATM_CURRENT'}

        if required_cols.issubset(self.df.columns):
            features_df = pd.DataFrame()
            for frame in self.time_frames:

                filt  = (self.df['MONTHS_BALANCE'] >= -frame) & (self.df['MONTHS_BALANCE'] < 0)


                monthly_df = self.df.loc[filt].copy()

                monthly_usage = (monthly_df.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])['CNT_DRAWINGS_ATM_CURRENT'].sum().to_frame())

                feature_df = monthly_usage.groupby('SK_ID_CURR')['CNT_DRAWINGS_ATM_CURRENT'].mean().to_frame(f'AVG_ATM_WITHDRAWAL_FREQ_{frame}M')
                features_df  = self._safe_join(features_df,feature_df)
                


            return features_df
        else:
            logger.debug(f'{required_cols} are not present in dataframe')

#-----------------------------------------------
    def _extract_pos_utilization_features(self):

        ''' Extract maximum POS spending utilization features over multiple time windows
                from the Credit Card Balance dataset.

            Features Transformed:
            - MAX_POS_SPEND_RATIO_XM: Maximum ratio of POS spending to total credit limit over the last X months.
            - MAX_POS_TO_TOTAL_DRAW_RATIO_XM:  Maximum ratio of POS spending to total monthly spend over the last X months.

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_DRAWINGS_POS_CURRENT','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_CURRENT'}
        if required_cols.issubset(self.df.columns):
            
            features_df = pd.DataFrame()
            
            monthly_df  = self.monthly_credit_agg

        
            for frame in self.time_frames:
                

                filt  = (monthly_df['MONTHS_BALANCE'] >= -frame) & (monthly_df['MONTHS_BALANCE'] < 0)

                monthly_frame = monthly_df.loc[filt].copy()

                monthly_frame['POS_SPEND_RATIO_PER_MONTH'] = np.where(
                            monthly_frame['AMT_CREDIT_LIMIT_ACTUAL_TOTAL'] > 0,
                            monthly_frame['AMT_DRAWINGS_POS_CURRENT_TOTAL'] / monthly_frame['AMT_CREDIT_LIMIT_ACTUAL_TOTAL'],
                            np.nan
                        )  
                monthly_frame['POS_TO_TOTAL_DRAW_RATIO'] = np.where(
                            monthly_frame['AMT_DRAWINGS_CURRENT_TOTAL'] > 0,
                            monthly_frame['AMT_DRAWINGS_POS_CURRENT_TOTAL'] / monthly_frame['AMT_DRAWINGS_CURRENT_TOTAL'],
                            np.nan
                        )  
                
                feature_df_1 = monthly_frame.groupby(by='SK_ID_CURR')['POS_TO_TOTAL_DRAW_RATIO'].max().to_frame(f'MAX_POS_TO_TOTAL_DRAW_RATIO_{frame}M')
                
                feature_df_2 = monthly_frame.groupby(by='SK_ID_CURR')['POS_SPEND_RATIO_PER_MONTH'].max().to_frame(f'MAX_POS_SPEND_RATIO_{frame}M')

                feature_df_list = [feature_df_1,feature_df_2]
                frame_features  = pd.concat(feature_df_list,axis=1)

                if features_df.empty:
                    features_df = frame_features
                else:
                    features_df = pd.concat([features_df, frame_features], axis=1)
 
            return features_df
        
        else:
            logger.debug(f"{required_cols} column are not present in the DataFrame")
    #-----------------------------------------------------------------------
    def _extract_payment_behavior_features(self):
        ''' Extracts payment behavior features from the Credit Card Balance dataset over multiple time windows.

            time_frames = [3,6,12,24] Months

            Features Transformed:
            - MAX_AMT_PAYMENT_MIN_INST_RATIO_XM:  Max Average ratio of payments made to minimum required payments over the last X months
            - UNDERPAYMENT_RATIO_XM:  Ratio of months where actual payment < minimum required payment  over the last X months
            - PAYMENT_VOLATILITY_STD_XM: Standard deviation of payment-to-mini  mum-installment ratio over the last X months
            - PAYMENT_TO_BALANCE_RATIO_XM: Max payment-to-balance ratio over last X months
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_INST_MIN_REGULARITY','AMT_PAYMENT_CURRENT','AMT_BALANCE'}
        if required_cols.issubset(self.df.columns):

            features_df = pd.DataFrame()
            monthly_df  = self.monthly_credit_agg

        
            for frame in self.time_frames:

                filt  = (monthly_df['MONTHS_BALANCE'] >= -frame) & (monthly_df['MONTHS_BALANCE'] < 0)
                monthly_frame = monthly_df.loc[filt].copy()

                # payment-to-minimum-installment ratio
                monthly_frame['AMT_PAYMENT_MIN_INST_RATIO'] = np.where(
                        monthly_frame['AMT_INST_MIN_REGULARITY_TOTAL'] > 0,
                        monthly_frame['AMT_PAYMENT_CURRENT_TOTAL'] / monthly_frame['AMT_INST_MIN_REGULARITY_TOTAL'],
                        np.nan
                            )
                monthly_frame['PAYMENT_TO_BALANCE_RATIO'] = np.where(
                        monthly_frame['AMT_BALANCE_TOTAL'] > 0,
                        monthly_frame['AMT_PAYMENT_CURRENT_TOTAL'] / monthly_frame['AMT_BALANCE_TOTAL'],
                        np.nan
                            )
                #underpayment flag
                monthly_frame['UNDERPAYMENT_FLAG'] = (monthly_frame['AMT_PAYMENT_CURRENT_TOTAL'] < monthly_frame['AMT_INST_MIN_REGULARITY_TOTAL']).astype(int)
                
                # max payment / min installmetn ratio
                feature_df_1=  monthly_frame.groupby(by='SK_ID_CURR')['AMT_PAYMENT_MIN_INST_RATIO'].max().to_frame(f'MAX_AMT_PAYMENT_MIN_INST_RATIO_{frame}M')

                #underpayment ratio
                feature_df_2=  monthly_frame.groupby(by='SK_ID_CURR')['UNDERPAYMENT_FLAG'].mean().to_frame(f'UNDERPAYMENT_RATIO_{frame}M')
                
                # payment volatility (standard deviation)
                feature_df_3=  monthly_frame.groupby(by='SK_ID_CURR')['AMT_PAYMENT_MIN_INST_RATIO'].std().to_frame(f'PAYMENT_VOLATILITY_STD_{frame}M')
                
                # PAYMENT_TO_BALANCE_RATIO
                feature_df_4 = monthly_frame.groupby('SK_ID_CURR')['PAYMENT_TO_BALANCE_RATIO'].max().to_frame(f'MAX_PAYMENT_BALANCE_RATIO_{frame}M')


                feature_df_list = [feature_df_1,feature_df_2,feature_df_3,feature_df_4]
                frame_features  = pd.concat(feature_df_list,axis=1)

                if features_df.empty:
                    features_df = frame_features
                else:
                    features_df = pd.concat([features_df, frame_features], axis=1)

            return features_df
        
        else:
            logger.debug(" MONTHS_BALANCE', 'AMT_INST_MIN_REGULARITY','AMT_PAYMENT_CURRENT' column are not present in the DataFrame")
        
#-----------------------------------------------------------------------

    def _extract_credit_utilization_ratios(self):

        ''' Extract features from the, 'AMT_RECEIVABLE_PRINCIPAL','AMT_CREDIT_LIMIT_ACTUAL','AMT_TOTAL_RECEIVABLE' in the Credit Balance dataset.

            Features Transformed:
            - MAX_PRINCIPAL_UTILIZATION_RATIO_{X}M: MAX Average ratio of receivable principal to credit limit in las x months.
           
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = { 'AMT_RECEIVABLE_PRINCIPAL','AMT_CREDIT_LIMIT_ACTUAL','MONTHS_BALANCE'}
        if required_cols.issubset(self.df.columns):

            features_df = pd.DataFrame()
            monthly_df  = self.monthly_credit_agg


            for frame in self.time_frames:
                filt  = (monthly_df['MONTHS_BALANCE'] >= -frame) & (monthly_df['MONTHS_BALANCE'] < 0)
                monthly_frame = monthly_df.loc[filt].copy()
    

                #represents how much principal amount of their credit limit is currently used.
                monthly_frame['PRINCIPAL_UTILIZATION_RATIO'] = np.where(
                            monthly_frame['AMT_CREDIT_LIMIT_ACTUAL_TOTAL'] > 0,
                            monthly_frame['AMT_RECEIVABLE_PRINCIPAL_TOTAL'] / monthly_frame['AMT_CREDIT_LIMIT_ACTUAL_TOTAL'],
                            np.nan)

                feature_df = monthly_frame.groupby(by='SK_ID_CURR')['PRINCIPAL_UTILIZATION_RATIO'].max().to_frame(f'MAX_PRINCIPAL_UTILIZATION_RATIO_{frame}M')
                features_df = self._safe_join(features_df, feature_df)

            
            return features_df
        
        else:
            logger.debug(f"{required_cols} column are not present in the DataFrame")

#-------------------------------------------------------------------------        
  
    def _extract_worst_dpd_features_credit(self):
        '''  Create worst DPD (Days Past Due) features for multiple time frames 
            from the credit balance dataset
    
            Features Extracted:
            - WORST_DPD_CREDIT_XM based on time frame:[3, 6, 9, 12, 24, 36, 72, 96] M

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as index and WORST_DPD_CREDIT_XM features
                Missing values filled with the placeholder -99999

        '''
        if 'SK_DPD' in self.df.columns and 'MONTHS_BALANCE' in self.df.columns:
            time_frames = [3, 6, 9, 12, 24, 36, 72, 96]
            # empty dataframe to apend the feature into
            features_df = pd.DataFrame()

            for frame in time_frames:
                filt = self.df['MONTHS_BALANCE'] >= -frame
                temp = self.df.loc[filt].copy()
                feature_dpd = temp.groupby(by='SK_ID_CURR')['SK_DPD'].max().to_frame(f'WORST_DPD_CREDIT_{frame}M')
                
                features_df = self._safe_join(features_df, feature_dpd)


            features_df = features_df.sort_index()
            return features_df
        else:
            logger.debug('SK_DPD : column is not present in the DataFrame')

    def _extract_worst_dpd_def_features_credit(self):
        '''Create severe DPD (Days Past Due with tolerance) features for multiple time frames 
            from the Credit balance dataset.

            Features Extracted:
            - WORST_DPD_DEF_CREDIT_XM for time frames [3, 6, 9, 12, 24, 36, 72, 96]

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as index and WORST_DPD_DEF_POS_CASH_XM features
                Missing values filled with the placeholder -99999
            '''
        if 'SK_DPD_DEF' in self.df.columns and 'MONTHS_BALANCE' in self.df.columns:
            time_frames = [3, 6, 9, 12, 24, 36, 72, 96]
            # empty dataframe to apend the feature into
            features_df = pd.DataFrame()

            for frame in time_frames:
                filt = self.df['MONTHS_BALANCE'] >= -frame
                temp = self.df.loc[filt].copy()
                feature_dpd = temp.groupby(by='SK_ID_CURR')['SK_DPD_DEF'].max().to_frame(f'WORST_DPD_DEF_CREDIT_{frame}M')
                
                features_df = self._safe_join(features_df, feature_dpd)

            features_df = features_df.sort_index()
            return features_df
        else:
            logger.debug('MONTHS_BALANCE and SK_DPD_DEF: column is not present in the DataFrame')
                  

#-------------------------------------------------------------------------        

      
    def add_features_main(self,main_df):

        '''Extract and Create feature from Credit card balance dataset and append in Main Dataframe'''   
        try:
            self.feature_extractors  = [
                self._extract_credit_utilization_features,
                self._extract_credit_usage_features,
                self._extract_atm_cash_usage_features,
                self._extract_avg_atm_withdrawal_frequency,
                self._extract_pos_utilization_features,
                self._extract_payment_behavior_features,
                self._extract_credit_utilization_ratios,
                self._extract_worst_dpd_features_credit,
                self._extract_worst_dpd_def_features_credit
            ]

           
            for extractor in self.feature_extractors:
                features_df = extractor()
                features_cols = features_df.columns.to_list()

                # log the method is running
                method_name = extractor.__name__
                logger.info(f"Running {self.__class__.__name__}.{method_name}")

                main_df = main_df.merge(features_df,on='SK_ID_CURR',how='left')
                main_df[features_cols] = main_df[features_cols].fillna(PLACEHOLDER)

            logger.info("Aggregated features from Credit card balance dataframe successfully merged into the main  Application dataframe.")

            return main_df
         
        except Exception as e:
            logger.exception("Error while adding credit card balance features")
            raise MyException(e,sys,logger)

      
class DataTransformation:
     
    def __init__(self,data_transformation_config:DataTransformationConfig,data_ingestion_config:DataIngestionConfig):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_config = data_ingestion_config 
        self.main_df_path =  os.path.join(self.data_ingestion_config.artifact_raw_dir,r'application_data.csv')

        self.output_dir = self.data_ingestion_config.artifact_interim_dir
        self.output_path = os.path.join(
            self.output_dir,
            "main_df_transformed.csv"
        )
        self.feature_transformers =  [
            BureauBalanceTransformation,
                        BureauTransformer,
                        InstallmentsPaymentsTransformation,
                        PosCashBalanceTransformation,
                        PreviousApplicationsTransformation,
                        CreditBalanceTransformation]
        
    def _is_data_validated(self):
        '''check and load status of the data validation from yaml file
            
            return:
                True | False : status of data validation
        '''

        file = read_yaml_file(self.data_transformation_config.data_validation_yaml,logger)
        return  file.get('is_data_validated', False)
    
    
    def _load_and_prepare_main_df(self) -> pd.DataFrame:
        """
        Load and preprocess the main application dataframe.
        """
        logger.info("Loading and preprocessing main application dataframe")

        transformer = ApplicationDfTransformer(
            self.main_df_path,
            ApplicationDfConfig()
        )

        return transformer.run_preprocessing_steps()
    def _save_transformed_data(self, df: pd.DataFrame) -> None:
        """
        Save transformed dataframe artifact/interim.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        df.to_csv(self.output_path, index=False)

        logger.info(f"Transformed dataframe saved at: {self.output_path}")

   
    def _apply_feature_transformations(self, main_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sequentially apply all features from  transformation classes
        to the main dataframe.
        """
        for transformer_cls in self.feature_transformers:
            logger.info(f"Applying {transformer_cls.__name__}")

            transformer = transformer_cls(
                self.data_transformation_config,
                self.data_ingestion_config
            )
            main_df = transformer.add_features_main(main_df)
            del transformer
            gc.collect()


        return main_df 
       
    def run(self):
        if self._is_data_validated():
            logger.info('DATA IS VALIDATED')
            try:
                main_df = self._load_and_prepare_main_df()

                final_df = self._apply_feature_transformations(main_df)
                self._save_transformed_data(final_df)
            
                logger.info('DATA TRANSFORMATION COMPLETED  SUCCESSFULLY')

            except Exception as e:
                raise MyException(e,sys,logger)
        else:
            logger.error("Data validation failed. Transformation Didnt happened.")

 
if __name__ =='__main__':
    
    data_transformation_config = DataTransformationConfig()
    data_ingestion_config = DataIngestionConfig()

    data_transformation = DataTransformation(
        data_transformation_config=data_transformation_config,
        data_ingestion_config=data_ingestion_config
    )

    data_transformation.run()


   
