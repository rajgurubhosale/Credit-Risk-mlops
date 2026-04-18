import os
import sys
import numpy as np
import pandas as pd
from abc import ABC,abstractmethod
from src.utils.main_utils import read_yaml_file
from src.logger import *
from src.exception import *        
import gc
from src.constants.artifacts_paths import *
from src.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact,DataValidationArtifact


#both numerator and denominator missing placeholder
DN_MISSING_PLACEHOLDER = - 99999
#numerator missing
N_MISSING_PLACEHOLDER = - 88888
#denominator missing
D_MISSING_PLACEHOLDER = - 77777
# missing value placeholder 
PLACEHOLDER = - 99999
# loan but the data is missing
DPD_LOAN_DATA_MISSING  = - 66666    


logger = config_logger('03_data_transformation')


class RatioFeatureMixin:
    def _create_ratio_feature(self, df, numerator, denominator, feature_name):

        temp = df[['SK_ID_CURR', numerator, denominator]].copy()


        temp[feature_name] = np.select(
            condlist=[
                temp[numerator].isna() & temp[denominator].isna(),
                temp[denominator].isna(),
                temp[numerator].isna()
            ],
            choicelist=[
                DN_MISSING_PLACEHOLDER,
                D_MISSING_PLACEHOLDER,
                N_MISSING_PLACEHOLDER
            ],
            default=np.where(
                temp[denominator] == 0,
                np.nan,
                temp[numerator] / temp[denominator]
            )
        )

        return temp[['SK_ID_CURR', feature_name]]

class ApplicationDfTransformer(RatioFeatureMixin):
    '''basic preprocessing of application main dataset
        - preprocess df
        - create days to year features
        - handle place holder values such as xNA/XAP 
     '''
    def __init__(self,main_df_path:str,data_indestion_artifact:DataIngestionArtifact):
        '''load the dataset in self.main_df
        
            args:
                main_df_path: application dataset path
        
        '''
        self.data_ingestion_artifact = data_indestion_artifact
        self.main_df = pd.read_csv(main_df_path)

    def _create_features_main(self,numerator,denominator,feature_name):
        
         self.main_df[feature_name] = np.select(
            condlist=[
                self.main_df[numerator].isna() & self.main_df[denominator].isna(),
                self.main_df[denominator].isna(),
                self.main_df[numerator].isna()
            ],
            choicelist=[
                DN_MISSING_PLACEHOLDER,
                D_MISSING_PLACEHOLDER,
                N_MISSING_PLACEHOLDER
            ],
            default=np.where(
                self.main_df[denominator] == 0,
                np.nan,
                self.main_df[numerator] / self.main_df[denominator]
            )
        )
          
    def _preprocessing(self):
        '''
        simplifying values in Application Df
        '''
        try:
            simplify_feature_values = {
                'NAME_EDUCATION_TYPE':{'Secondary / secondary special':'Secondary education'},
                'NAME_FAMILY_STATUS':{'Single / not married': 'Single'},
                'NAME_HOUSING_TYPE':{ 'House / apartment': 'Owned'}
                }
            for col, mapping in simplify_feature_values.items():
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
            
            days_to_years_mapping = {
                    "DAYS_BIRTH": "YEARS_AGE",
                    "DAYS_EMPLOYED": "YEARS_EMPLOYED",
                    "DAYS_REGISTRATION": "YEARS_REGISTRATION",
                    "DAYS_ID_PUBLISH": "YEARS_ID_PUBLISH"
            }
            
            for col,trasform_col in days_to_years_mapping.items():
                
                if col in self.main_df.columns:
                    #converting the days feature to the years
                    self.main_df[col] = self.main_df[col].replace({365243:np.nan})
                    self.main_df[trasform_col] =  (-self.main_df[col] / 365).round(2)    

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
            - 1000.669983 : -99999 PLACEHOLDER
        '''
        try:
            # feature wise handling local placeholders
            
            placeholders = {
                'local_placeholder': 
                {
                    'YEARS_EMPLOYED':{-1000.67:-99999},
                },
                
                'global_placeholders':{"XNA": 'Missing', "XAP": 'Missing', "Unknown": 'Missing'}
            }

            for col , mapping in placeholders['local_placeholder'].items():
                if col in self.main_df.columns:
                    self.main_df[col] = self.main_df[col].replace(mapping)
                else:
                    logger.debug(f'{col} : column is not present in the DataFrame')

            # global placeholders
            self.main_df = self.main_df.replace(placeholders['global_placeholders'])
            
        except Exception as e:

            raise MyException(e,sys,logger)
    def _extract_credit_income_ratio(self):
        ''' Total credit amount to client income ratio.
            Features Transformed:
            - CREDIT_INCOME_RATIO: AMT_CREDIT / AMT_INCOME_TOTAL
            
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        ''' 
        self._create_features_main('AMT_CREDIT','AMT_INCOME_TOTAL','CREDIT_INCOME_RATIO')
        
    def _extract_annuity_income_ratio(self):
        ''' Monthly annuity to client income ratio.
            Measures monthly repayment burden relative to client income
            Features Transformed:
            - ANNUITY_INCOME_RATIO: AMT_ANNUITY / AMT_INCOME_TOTAL
        '''
        self._create_features_main('AMT_ANNUITY','AMT_INCOME_TOTAL','ANNUITY_INCOME_RATIO')
        
    def _extract_goods_credit_ratio(self):
        '''  Goods price to credit amount ratio.
            Measures the proportion of the loan amount that is backed by the goods price.
            Features Transformed:
            - GOODS_CREDIT_RATIO: AMT_GOODS_PRICE / AMT_CREDIT  
           
        '''
        self._create_features_main('AMT_GOODS_PRICE','AMT_CREDIT','GOODS_CREDIT_RATIO')
    
    def _extract_annuity_credit_ratio(self):
        '''  Annuity to credit amount ratio.
            Measures the proportion of the total credit that is paid as annuity
            Features Transformed:
            - ANNUITY_CREDIT_RATIO: AMT_ANNUITY / AMT_CREDIT
        '''
        self._create_features_main('AMT_ANNUITY','AMT_CREDIT','ANNUITY_CREDIT_RATIO')

    def _map_organization_to_group(self):
        """
        Consolidate ORGANIZATION_TYPE into broader organizational stability groups.
        for Reduce high-cardinality noise from ORGANIZATION_TYPE and
        Capture employment stability signal for credit risk modeling        
        
        Features Transformed:
        - ORG_GROUP  :ORG_STABLE,ORG_PRIVATE,ORG_UNSTABLE,ORG_OTHER
        
        """

        ORG_STABLE = [
            "Government", "School", "University", "Medicine", "Police",
            "Military", "Bank", "Insurance", "Security Ministries",
            "Electricity", "Postal"
        ]

        ORG_PRIVATE = [
            "Business Entity Type 1", "Business Entity Type 2", "Business Entity Type 3",
            "Industry: type 1", "Industry: type 2", "Industry: type 3",
            "Industry: type 4", "Industry: type 5", "Industry: type 7",
            "Industry: type 9", "Industry: type 11",
            "Trade: type 1", "Trade: type 2", "Trade: type 3",
            "Trade: type 6", "Trade: type 7",
            "Transport: type 1", "Transport: type 2", "Transport: type 4",
            "Telecom", "Services", "Housing", "Mobile", "Security"
        ]

        ORG_UNSTABLE = [
            "Self-employed", "Construction", "Agriculture", "Restaurant",
            "Cleaning", "Realtor", "Advertising",
            "Industry: type 8", "Industry: type 13", "Transport: type 3"
        ]

        ORG_OTHER = [
            "Religion", "Culture", "Emergency", "Legal Services",
            "Industry: type 6", "Industry: type 10", "Industry: type 12",
            "Trade: type 4", "Trade: type 5"
        ]

        self.main_df["ORG_GROUP"] = "ORG_OTHER"

        self.main_df.loc[self.main_df["ORGANIZATION_TYPE"].isin(ORG_STABLE), "ORG_GROUP"] = "ORG_STABLE"
        self.main_df.loc[self.main_df["ORGANIZATION_TYPE"].isin(ORG_OTHER), "ORG_GROUP"] = "ORG_OTHER"
        self.main_df.loc[self.main_df["ORGANIZATION_TYPE"].isin(ORG_PRIVATE), "ORG_GROUP"] = "ORG_PRIVATE"
        self.main_df.loc[self.main_df["ORGANIZATION_TYPE"].isin(ORG_UNSTABLE), "ORG_GROUP"] = "ORG_UNSTABLE"
        self.main_df.loc[self.main_df["ORGANIZATION_TYPE"] == "XNA", "ORG_GROUP"] = "MISSING"

    def _extract_document_provided_flag(self):
        """
        Create a binary indicator for whether the client submitted at least one document.
         Reduces 20 sparse FLAG_DOCUMENT_* variables into a single feature

        Features Transformed:
        - DOCUMENT_PROVIDED_FLAG (binary)
            * 0 : No documents submitted
            * 1 : At least one document submitted
        """

        DOC_FLAGS = [
            "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5",
            "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9",
            "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
            "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17",
            "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21"
        ]

        self.main_df["DOCUMENT_PROVIDED_FLAG"] = self.main_df[DOC_FLAGS].sum(axis=1)

        # Bin into binary indicator
        self.main_df["DOCUMENT_PROVIDED_FLAG"] = (
            self.main_df["DOCUMENT_PROVIDED_FLAG"] > 0
        ).astype(int)
        
    
    def _bin_education_level(self):
        """
        Bins NAME_EDUCATION_TYPE into ordered education levels.

        Mapping:
        - Higher education + Academic degree -> Higher education
        - Secondary / secondary special + Incomplete higher -> Secondary
        - Lower secondary -> Lower secondary
        """
        education_map = {
            "Higher education": "Higher education",
            "Academic degree": "Higher education",
            "Secondary / secondary special": "Secondary",
            "Incomplete higher": "Secondary",
            "Lower secondary": "Lower secondary"
        }

        self.main_df["NAME_EDUCATION_TYPE"] = self.main_df["NAME_EDUCATION_TYPE"].map(education_map)
        
    def _handle_missing_family_status(self):
        """
        Cleans NAME_FAMILY_STATUS by mapping unknown values
        to an explicit MISSING category.
        """

        self.main_df["NAME_FAMILY_STATUS"] = (
            self.main_df["NAME_FAMILY_STATUS"]
            .replace("Unknown", "MISSING")
        )
        
    def _map_occupation_to_group(self):
        """
        Groups OCCUPATION_TYPE into consolidated OCCUPATION_GROUP
        based on skill level and job stability.
        """
        LOW_SKILL = [
            "Low-skill Laborers", "Drivers", "Waiters/barmen staff",
            "Security staff", "Laborers", "Cleaning staff", "Cooking staff"
        ]

        SERVICE = [
            "Sales staff", "Private service staff", "Realty agents"
        ]

        SKILLED_PRO = [
            "Core staff", "High skill tech staff", "IT staff",
            "Medicine staff", "Accountants", "HR staff", "Secretaries"
        ]

        MANAGERS = ["Managers"]

        self.main_df["OCCUPATION_GROUP"] = "MISSING"

        self.main_df.loc[self.main_df["OCCUPATION_TYPE"].isin(LOW_SKILL), "OCCUPATION_GROUP"] = "LOW_SKILL"
        self.main_df.loc[self.main_df["OCCUPATION_TYPE"].isin(SERVICE), "OCCUPATION_GROUP"] = "SERVICE"
        self.main_df.loc[self.main_df["OCCUPATION_TYPE"].isin(SKILLED_PRO), "OCCUPATION_GROUP"] = "SKILLED_PRO"
        self.main_df.loc[self.main_df["OCCUPATION_TYPE"].isin(MANAGERS), "OCCUPATION_GROUP"] = "MANAGERS"
        self.main_df.drop(columns=['OCCUPATION_TYPE'],inplace=True)

    def _encode_binary_flag_features(self):
        ''' Convert binary flag features from categorical (1/0) to numeric (Y/N).
            for binning purpose
            maping: 1:Y,0:N
        
        Features Transformed:
        - ['FLAG_WORK_PHONE','FLAG_EMP_PHONE','FLAG_PHONE','FLAG_EMAIL']
        
        '''
        features = ['FLAG_WORK_PHONE','FLAG_EMP_PHONE','FLAG_PHONE','FLAG_EMAIL']
        for feature in features:
            self.main_df[feature+'_ENCODED'] = self.main_df[feature].map({1:'Y',0:'N'})
        
    def _extract_social_any_default_flag(self):
        """
        Creates a binary indicator capturing whether the client has
        any default observed in their social circle within 30 or 60 days.
        Features Transformed:
        - SOCIAL_ANY_DEFAULT

        """
        self.main_df["SOCIAL_ANY_DEFAULT"] = (
            (self.main_df["DEF_30_CNT_SOCIAL_CIRCLE"] > 0) |
            (self.main_df["DEF_60_CNT_SOCIAL_CIRCLE"] > 0)
        ).astype(int)
        
        

    def _create_remaining_features(self):
        ''' this method is created while working on model this feature are 
            bruitforce feature tried so thery are in one place
        '''
        
        # EXT SOURCE FEATURE
        #THE WEIGHTS are obtained by training the log reg model with 3 ext source featur and use here

        weights  = [1.9405, 2.4851, 2.7281]

        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        ext_df   = self.main_df[ext_cols]

        active_weights  = ext_df.notna() * weights        # zero-out weight where NaN
        effective_denom = active_weights.sum(axis=1)      # only available weights sum

        self.main_df['EXT_SOURCE_WEIGHTED'] = (
            (ext_df.fillna(0) * weights).sum(axis=1) / effective_denom
        )
            
        self.main_df['EXT_SOURCE_X_INCOME']  = self.main_df['EXT_SOURCE_WEIGHTED'] * self.main_df['AMT_INCOME_TOTAL']
        self.main_df['EXT_SOURCE_X_AGE']  = self.main_df['EXT_SOURCE_WEIGHTED'] * self.main_df['YEARS_AGE']
        self.main_df['CREDIT_EXT_SOURCE_PRODUCT']  = self.main_df['EXT_SOURCE_WEIGHTED'] * self.main_df['AMT_CREDIT']

        self.main_df['EXT_SOURCE_MIN'] = (self.main_df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1))
        self.main_df['EXT_SOURCE_MAX'] = (self.main_df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].max(axis=1))
        self.main_df['EXT_SOURCE_RANGE'] = (self.main_df['EXT_SOURCE_MAX']- self.main_df['EXT_SOURCE_MIN'])
        
        # more than 2> merge good feature
        self.main_df['EXT_SOURCE_MISSING_CNT'] = self.main_df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].isnull().sum(axis=1)
        self.main_df['CREDIT_GOODS_DIFF_AMT'] = (self.main_df['AMT_CREDIT']- self.main_df['AMT_GOODS_PRICE'])
    
    
        self._create_features_main('CNT_CHILDREN','CNT_FAM_MEMBERS','CHILDREN_RATIO')               
        


        self._create_features_main('AMT_CREDIT','YEARS_AGE','CREDIT_PER_YEAR_AGE')               

        
        self._create_features_main('AMT_INCOME_TOTAL','YEARS_AGE','INCOME_TO_AGE_RATIO')               
    
        
        self._create_features_main('YEARS_EMPLOYED','YEARS_AGE','EMPLOYMENT_TO_AGE_RATIO')               
        
        self.main_df['EMPLOYMENT_GAP'] =self.main_df['YEARS_AGE']  - self.main_df['YEARS_EMPLOYED']


        self._create_features_main('YEARS_REGISTRATION','YEARS_AGE','REGISTRATION_TO_AGE_RATIO')       

        self._create_features_main('YEARS_ID_PUBLISH','YEARS_AGE','ID_TO_AGE_RATIO')       
                
        self._create_features_main('AMT_INCOME_TOTAL','CNT_FAM_MEMBERS','INCOME_PER_PERSON')     
          
        # denominator needs 0+1 scaled
        self.main_df['ADD_1_CNT_CHILDERN'] = 1 + self.main_df['CNT_CHILDREN']
        self._create_features_main('AMT_INCOME_TOTAL','ADD_1_CNT_CHILDERN','INCOME_PER_CHILD')       
        self._create_features_main('AMT_ANNUITY','ADD_1_CNT_CHILDERN','ANNUITY_PER_CHILD')       


        self._create_features_main('AMT_CREDIT','CNT_FAM_MEMBERS','CREDIT_PER_PERSON')        
        self._create_features_main('DAYS_LAST_PHONE_CHANGE','YEARS_AGE','PHONE_TO_AGE_RATIO')

        # good feature
        self.main_df['REGION_RATING_DIFF'] =   self.main_df['REGION_RATING_CLIENT_W_CITY'] -   self.main_df['REGION_RATING_CLIENT']
        self._create_features_main('AMT_ANNUITY','YEARS_AGE','ANNUITY_TO_AGE_RATIO')

        bureau_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR'
        ]

        self.main_df['TOTAL_BUREAU_REQUESTS'] = self.main_df[bureau_cols].sum(axis=1)
        
        #Housing Aggregation Features
        self._create_features_main('LIVINGAREA_AVG','TOTALAREA_MODE','LIVINGAREA_RATIO')
        self._create_features_main('NONLIVINGAREA_AVG','TOTALAREA_MODE','NONLIVING_RATIO')
        self._create_features_main('FLOORSMAX_AVG','FLOORSMIN_AVG','FLOOR_RATIO')

        
        # good feature 
        self.main_df['FINANCIAL_STRESS_SCORE'] = (
            self.main_df['CREDIT_INCOME_RATIO']
            + self.main_df['ANNUITY_INCOME_RATIO']
            )

        self.main_df['BURDEN_RISK_INTERACTION'] = (
            self.main_df['CREDIT_INCOME_RATIO'] *
            self.main_df['EXT_SOURCE_WEIGHTED']
        )
        
        
        self._create_features_main('DEF_30_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','SOCIAL_OBS_DEF_RATIO_30')
        self._create_features_main('DEF_60_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','SOCIAL_OBS_DEF_RATIO_60')
        
        phone_cols = ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL']
        self.main_df['PHONE_FLAG_COUNT'] = self.main_df[phone_cols].sum(axis=1)
        
        self.main_df['INCOME_BY_ORG_TYPE'] = (self.main_df.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].transform('median'))
        self.main_df.drop(columns=['ORGANIZATION_TYPE'],inplace=True)
        

        # Difference from organization median
        self.main_df['INCOME_DIFF_FROM_ORG_MEDIAN'] = (
            self.main_df['AMT_INCOME_TOTAL'] -
            self.main_df['INCOME_BY_ORG_TYPE']
        )
        
        
    def run_preprocessing_steps(self):

        '''run all preprocessing steps in sequence

            return:
                self.main_df: returns the dataframe after all preprocessing
            '''
        try:
            methods = [
            self._preprocessing,
            self._convert_days_to_years,
            self._replace_placeholders,
            self._extract_credit_income_ratio,
            self._extract_annuity_income_ratio,
            self._extract_goods_credit_ratio,
            self._extract_annuity_credit_ratio,
            self._map_organization_to_group,
            self._extract_document_provided_flag,
            self._bin_education_level,
            self._handle_missing_family_status,
            self._map_occupation_to_group,
            self._encode_binary_flag_features,
            self._extract_social_any_default_flag,
            self._create_remaining_features
            ]
            class_name = self.__class__.__name__
            for method in methods:
                method_name = method.__name__
                logger.info(f"Current Method Running: {class_name}.{method_name}")        
                method()

            logger.info('Application DataFrame preprocessing done successfully')

        except Exception as e:
            raise MyException(e,sys,logger)

        return self.main_df
    



class BaseTransformer(ABC):
    '''combine all the diff dataset into one single dataframe for analysis and model pipeline'''

    def __init__(self,data_transformation_artifact:DataTransformationArtifact,data_indestion_artifact:DataIngestionArtifact):

        self.data_transformation_artifact = data_transformation_artifact
        self.data_indestion_artifact = data_indestion_artifact

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load data and optimize memory usage by downcasting numeric types."""
        
        try:
            # Load data
            df = pd.read_csv(data_path)

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
            raise Exception(f"Error loading data from {data_path}: {e}")
        

    @abstractmethod
    def add_features_main(self,main_df=None):
        '''create features and append in main dataframe'''
        pass
    

  

class BureauBalanceTransformation(BaseTransformer):
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, data_indestion_artifact: DataIngestionArtifact):
        super().__init__(data_transformation_artifact, data_indestion_artifact)

        self.bureau_balance = self.load_data(
            self.data_indestion_artifact.bureau_balance,
        )
        
        self.bureau = self.load_data(
            self.data_indestion_artifact.bureau_data,
            
        )
        
        dpd_map = {'X':np.nan,'C':0,'1':1,'2':2,'3':3,'4':4,'5':5}
        self.bureau_balance['STATUS'] = self.bureau_balance['STATUS'].map(dpd_map)


    def _extract_worst_dpd_features(self):
        '''  Create worst DPD (Days Past Due) features for multiple time frames 
            from the bureau_balance dataset.
    
            Features Extracted:
            - WORST_DPD_ based on time frame:[3, 6, 9, 12, 24, 36, 72, 96] M

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as index and WORST_DPD_BUREAU_XM features
                Missing values filled with the placeholder -66666

        '''
        if 'STATUS' in self.bureau_balance.columns:

            time_frames = [3, 6, 9, 12, 24, 36, 72, 96]
            # empty dataframe to apend the feature into
            features = []

            for frame in time_frames:
                filt = self.bureau_balance['MONTHS_BALANCE'] >= -frame
                temp = self.bureau_balance.loc[filt]
                feature_dpd = temp.groupby(by='SK_ID_BUREAU')['STATUS'].max().to_frame(f'BB_WORST_DPD_{frame}M')
                features.append(feature_dpd)

            features = pd.concat(features, axis=1).reset_index()
            
            features = features.fillna(DPD_LOAN_DATA_MISSING)

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
                Missing values filled with the placeholder -66666

        '''
        if 'STATUS' in self.bureau_balance.columns:
                        
            time_frames = [12,24,36,72]
            # empty dataframe to apend the feature into
            features = []

            for frame in time_frames:
                filt = (self.bureau_balance['MONTHS_BALANCE'] >= -frame) & (self.bureau_balance['STATUS'] >= 3)
                temp = self.bureau_balance.loc[filt]
                feature_dpd = temp.groupby(by='SK_ID_BUREAU')['STATUS'].max().to_frame(f'BB_SEVERE_DPD_{frame}M')
                features.append(feature_dpd)

            features = pd.concat(features, axis=1).reset_index()
            features = features.fillna(DPD_LOAN_DATA_MISSING)
            
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

            features = (temp.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max().to_frame('BB_RECENT_MONTH_OF_DPD').reset_index())

            bureau_agg = (
                self.bureau[['SK_ID_BUREAU', 'SK_ID_CURR']]
                .merge(features, on='SK_ID_BUREAU', how='left')
                .groupby('SK_ID_CURR')
                .max()
                .reset_index()
            )
            bureau_agg = bureau_agg.drop(columns='SK_ID_BUREAU')

            return bureau_agg.fillna(DPD_LOAN_DATA_MISSING)
        else:
            logger.debug('STATUS : column is not present in the DataFrame')
    
    def _extract_worst_dpd_active_loans(self):
        '''
        Extract worst DPD status among Active loans.

        Features Extracted:
        - BB_WORST_DPD_ACTIVE_LOANS:
            Worst STATUS among active bureau loans.

        Returns:
        - bureau_agg:
            DataFrame with SK_ID_CURR and BB_WORST_DPD_ACTIVE_LOANS feature
        '''

        if ('CREDIT_ACTIVE' in self.bureau.columns) and ('STATUS' in self.bureau_balance.columns):

            # Filter bureau for Active loans
            active_bureau = self.bureau[self.bureau['CREDIT_ACTIVE'] == 'Active']
            active_customers = active_bureau[['SK_ID_CURR']].drop_duplicates()
            active_ids = active_bureau['SK_ID_BUREAU']

            # Filter bureau_balance for those loans
            active_bb = self.bureau_balance[
                self.bureau_balance['SK_ID_BUREAU'].isin(active_ids)
            ]

            # Worst STATUS per bureau loan
            worst_status = (
                active_bb
                .groupby('SK_ID_BUREAU')['STATUS']
                .max()
                .to_frame()
                .reset_index()
            )

            # Merge back to bureau
            bureau_temp = self.bureau.merge(
                worst_status,
                on='SK_ID_BUREAU',
                how='left'
            )

            # Aggregate per customer
            bureau_agg = (
                bureau_temp
                .groupby('SK_ID_CURR')['STATUS']
                .max()
                .to_frame('BB_WORST_DPD_ACTIVE_LOANS')
                .reset_index()
            )

            result = active_customers.merge(bureau_agg, on='SK_ID_CURR', how='left')
            result['BB_WORST_DPD_ACTIVE_LOANS'] = result['BB_WORST_DPD_ACTIVE_LOANS'].fillna(DPD_LOAN_DATA_MISSING)  # -66666

            return result

        else:
            logger.debug(
                'CREDIT_ACTIVE or STATUS column is not present in the DataFrame'
            )

    def _extract_features(self):
        
        df_agg =  self.bureau_balance.groupby('SK_ID_BUREAU').agg(
            MONTHS_BALANCE_MAX=('MONTHS_BALANCE', 'max'),
            MONTHS_BALANCE_MEAN=('MONTHS_BALANCE', 'mean'),
            MONTHS_BALANCE_SUM=('MONTHS_BALANCE', 'sum')
        ).reset_index()

        # Step 2 : Merge with bureau to map to SK_ID_CURR
        df_agg_curr = self.bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(df_agg, on='SK_ID_BUREAU', how='left').groupby('SK_ID_CURR').max().reset_index()
        df_agg_curr = df_agg_curr.drop(columns='SK_ID_BUREAU')
        
        return df_agg_curr

    def add_features_main(self, main_df):
        """Extract and attach bureau-balance features to main_df"""
        try:
            features_extractors = [
                self._extract_worst_dpd_features,
                self._extract_severe_dpd_features,
                self._extract_month_recent_dpd,
                self._extract_worst_dpd_active_loans,
                self._extract_features
            ]

            for extractor in features_extractors:
                features_df = extractor()
                # log the method is running
                method_name = extractor.__name__
                logger.info(f"Current Method Running: {self.__class__.__name__}.{method_name}")        

               
                main_df = main_df.merge(
                    features_df,
                    on='SK_ID_CURR',
                    how='left'
                )
                

                new_cols = features_df.columns.drop('SK_ID_CURR')

                main_df[new_cols] = main_df[new_cols].fillna(PLACEHOLDER)

            logger.info(
                "Successfully merged Bureau Balance features into main dataframe."
            )
            del features_df
            gc.collect()
            
            return main_df

        except Exception as e:
            raise MyException(e, sys, logger)



class BureauTransformer(BaseTransformer,RatioFeatureMixin):
    '''Extracts and Transform features from the bureau data. And append in the main dataframe'''
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, data_indestion_artifact: DataIngestionArtifact):
        '''Load the bureau dataset and save in the self.df dataframe'''
        super().__init__(data_transformation_artifact, data_indestion_artifact)

        self.df = self.load_data(
            data_path = self.data_indestion_artifact.bureau_data,
        )
        

    def _extract_num_credit_currencies(self):

        '''  Count the number of different currencies a client has taken loans in.

            Features Transformed:
            - NUM_CREDIT_CURRENCIES: number of unique credit currencies the client had taken the loan

            Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''
        if 'CREDIT_CURRENCY' in self.df.columns:
            feature_df = self.df.groupby(by='SK_ID_CURR')['CREDIT_CURRENCY'].nunique().to_frame('B_NUM_CREDIT_CURRENCIES')

            # no currency -> 0
            #no data availability -> placeholder
            feature_df = feature_df.fillna(0)
            return feature_df.reset_index()

        else:
            logger.debug('CREDIT_CURRENCY : column is not present in the DataFrame')

    def _extract_num_active_credit_d(self):
        '''  Create num active credit features for multiple time frames 
            from the bureau dataset.
    
            Features Extracted:
            - NUM_ACTIVE_CREDIT_XD: based on time frame: [90, 180, 270, 360, 720,1080] Days


            Returns:
            - feature_df : 
                DataFrame with SK_ID_BUREAU as index and NUM_ACTIVE_CREDIT_XD features
                Missing values filled with the placeholder -99999

        '''
        if 'CREDIT_ACTIVE' in self.df .columns:
            

            
            time_frames = [90, 180, 270, 360, 720,1080]

            # empty dataframe to apend the feature into
            features_df = pd.DataFrame()


            for frame in time_frames:
                filt = self.df['DAYS_CREDIT'] >= -frame
                temp = self.df.loc[filt].copy()

                crosstab = pd.crosstab(temp['SK_ID_CURR'],temp['CREDIT_ACTIVE'])
                    

                active_credit = crosstab.get('Active', pd.Series(0, index=crosstab.index)).to_frame(f'B_NUM_ACTIVE_CREDIT_{frame}D')

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
            temp = self.df.loc[filt]
            features_df = temp.groupby(by='SK_ID_CURR')['DAYS_CREDIT'].max().to_frame('B_DAYS_SINCE_LAST_BAD_LOAN')
            
            # converting it into positive number so i can use the -99999 placeholder later for the null values
            features_df['B_DAYS_SINCE_LAST_BAD_LOAN'] = -features_df['B_DAYS_SINCE_LAST_BAD_LOAN']
            

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
            
            self.df['B_HAS_CREDIT_DAYS_OVERDUE'] = np.where(
            self.df['CREDIT_DAY_OVERDUE'].isna(), np.nan, (self.df['CREDIT_DAY_OVERDUE'] > 0).astype(int))

            # aggreagate
            features_df = self.df.groupby('SK_ID_CURR')['B_HAS_CREDIT_DAYS_OVERDUE'].max().to_frame()

            return features_df.reset_index()
        else:
            logger.debug('CREDIT_DAY_OVERDUE : column is not present in the DataFrame')


    def _extract_days_enddate(self):

        ''' Extract features from the 'DAYS_ENDDATE_FACT' and 'DAYS_CREDIT_ENDDATE' column in the bureau dataframe.
            
            Features Transformed:
            - B_AVG_REPAYMENT_DAYS_DIFF : Average diff in days between actual and scheduled credit end date for closed credits.
            - B_MAX_REPAYMENT_DAYS_DIFF : Max diff in days between actual and scheduled credit end date for closed credits.
            - B_MIN_REPAYMENT_DAYS_DIFF : min diff in days between actual and scheduled credit end date for closed credits.

             Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''
                
        if ('DAYS_ENDDATE_FACT' in self.df.columns) and ('DAYS_CREDIT_ENDDATE' in self.df.columns):

            filt = self.df['CREDIT_ACTIVE'] == 'Closed'
            new_df = self.df.loc[filt]
            new_df['REPAYMENT_DAYS_DIFF'] = (new_df['DAYS_ENDDATE_FACT'] - new_df['DAYS_CREDIT_ENDDATE'])
            
            features_df = new_df.groupby(by='SK_ID_CURR').agg( 
                B_AVG_REPAYMENT_DAYS_DIFF = ('REPAYMENT_DAYS_DIFF','mean'),
                B_MIN_REPAYMENT_DAYS_DIFF = ('REPAYMENT_DAYS_DIFF','min'),
                B_MAX_REPAYMENT_DAYS_DIFF =   ('REPAYMENT_DAYS_DIFF','max'))

            return features_df.reset_index()
        else:
            logger.debug('DAYS_ENDDATE_FACT, DAYS_CREDIT_ENDDATE: column is not present in the dataframe')


    def _amt_credit_max_overdue(self):

        ''' Extract flag has overdue and max amount overdue features from the 'AMT_CREDIT_MAX_OVERDUE' column
             in the bureau dataframe.
        
            Features Transformed:
            - FLAG_HAS_AMT_OVERDUE: flag if the customer had any amt overdue 1, 0
            - B_CREDIT_DAY_OVERDUE_SUM: sum of all overdue days across credits
            - B_CREDIT_DAY_OVERDUE_MAX: maximum overdue days across credits    
            - B_CREDIT_DAY_OVERDUE_MIN: min overdue days across credits    


            Returns:
                features_df(pd.DataFrame): dataframe with SK_ID_CURR Index and Transformed features
        '''

        if 'AMT_CREDIT_MAX_OVERDUE' in self.df.columns:      

            temp = self.df[['SK_ID_CURR', 'AMT_CREDIT_MAX_OVERDUE']]
  
            temp['B_FLAG_HAS_AMT_OVERDUE'] = (temp['AMT_CREDIT_MAX_OVERDUE'] > 0 ).astype(int)
            
            filt = temp['AMT_CREDIT_MAX_OVERDUE'].isnull()
            temp.loc[filt,'B_FLAG_HAS_AMT_OVERDUE'] = np.nan
            
            features_df = temp.groupby(by='SK_ID_CURR')['B_FLAG_HAS_AMT_OVERDUE'].max().to_frame()
          
            overdue_df = self.df.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].agg(
                        B_CREDIT_DAY_OVERDUE_SUM='sum',
                        B_CREDIT_DAY_OVERDUE_MIN='min',
                        B_CREDIT_DAY_OVERDUE_MAX='max'
                    )
            
            features_df = features_df.join(overdue_df,how='outer')
            return features_df.reset_index()
        else:
            logger.debug('AMT_CREDIT_MAX_OVERDUE: column is not present in the DataFrame')

    def _cnt_credit_prolong(self):
        ''' Extract features from the 'CNT_CREDIT_PROLONG' column in the bureau dataframe.
    
            Features Transformed:
            - MAX_CREDIT_PROLONG: max credit prolong for the customer

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'CNT_CREDIT_PROLONG' in self.df.columns:

            max_prolong = self.df.groupby(by='SK_ID_CURR')['CNT_CREDIT_PROLONG'].max().to_frame('B_MAX_CREDIT_PROLONG').reset_index()
            
            return max_prolong
           
        else:
            logger.debug('CNT_CREDIT_PROLONG: column is not present in the DataFrame')



    def _extract_features_amt_credit(self):
        ''' Extract features from the 'AMT_CREDIT_SUM_DEBT' and 'AMT_CREDIT_SUM' column in the bureau dataframe.
    
            Features Transformed:
            - B_RATIO_DEBT_TO_LOAN: ratio of total debt to total credit of customer (Active credits)
            - B_ACTIVE_DEBT_SUM: total debt of Active credits
            - B_ACTIVE_CREDIT_SUM: total credit of Active credits
            - B_AMT_CREDIT_SUM_DEBT_MEAN: mean debt across all credits
            - B_AMT_CREDIT_SUM_DEBT_SUM: total debt across all credits
            - B_AMT_CREDIT_SUM_DEBT_MAX: max debt from a single credit
            - B_TOTAL_UTIL_RATIO_DEBT_CREDIT:
            - B_TOTAL_ANNUITY_TO_DEBT: 
            - B_OVERDUE_TO_CREDIT_RATIO:

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''


        if ('AMT_CREDIT_SUM_DEBT' in self.df.columns) and ('AMT_CREDIT_SUM' in self.df.columns) and ('AMT_CREDIT_SUM_DEBT' in self.df.columns):
            
            
            
            
            active_df = self.df[self.df['CREDIT_ACTIVE']=='Active']

            active_sums = active_df.groupby('SK_ID_CURR').agg(
                        B_ACTIVE_DEBT_SUM=('AMT_CREDIT_SUM_DEBT', 'sum'),
                        B_ACTIVE_CREDIT_SUM=('AMT_CREDIT_SUM', 'sum')
                    ).reset_index()

            feature_df = self._create_ratio_feature(active_sums,'B_ACTIVE_DEBT_SUM','B_ACTIVE_CREDIT_SUM','B_RATIO_DEBT_TO_LOAN' )

            
            
            # Create aggregated debt features
            features_df_2 = self.df.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].agg(
                B_AMT_CREDIT_SUM_DEBT_MEAN='mean',
                B_AMT_CREDIT_SUM_DEBT_SUM='sum',
                B_AMT_CREDIT_SUM_DEBT_MIN='min',
                B_AMT_CREDIT_SUM_DEBT_MAX='max'
            ).reset_index()      
            
            agg_df = self.df.groupby('SK_ID_CURR').agg(
                        B_AMT_DEBT_SUM=('AMT_CREDIT_SUM_DEBT', 'sum'),
                        B_AMT_CREDIT_SUM=('AMT_CREDIT_SUM', 'sum'),
                        B_AMT_ANNUITY_SUM=('AMT_ANNUITY','sum'),
                        B_AMT_CREDIT_MAX_OVERDUE = ('AMT_CREDIT_MAX_OVERDUE','sum')
                    ).reset_index()
            
            
            features_df_3 = self._create_ratio_feature(agg_df,'B_AMT_DEBT_SUM','B_AMT_CREDIT_SUM','B_TOTAL_UTIL_RATIO_DEBT_CREDIT' )
            
            features_df_4 = self._create_ratio_feature(agg_df,'B_AMT_ANNUITY_SUM','B_AMT_DEBT_SUM','B_TOTAL_ANNUITY_TO_DEBT' )
            
            features_df_5 = self._create_ratio_feature(agg_df,'B_AMT_CREDIT_MAX_OVERDUE','B_AMT_CREDIT_SUM','B_OVERDUE_TO_CREDIT_RATIO' )
            
            features_df = active_sums.merge(feature_df[['SK_ID_CURR', 'B_RATIO_DEBT_TO_LOAN']], on='SK_ID_CURR', how='outer')            
            features_df = features_df.merge(features_df_2, on='SK_ID_CURR', how='outer')
            features_df = features_df.merge(features_df_3, on='SK_ID_CURR', how='outer')
            features_df = features_df.merge(features_df_4, on='SK_ID_CURR', how='outer')
            features_df = features_df.merge(features_df_5, on='SK_ID_CURR', how='outer')

            

            return features_df

        else:
            logger.debug('DEBT_TO_LOAN_RATIO ,AMT_CREDIT_SUM  AMT_CREDIT_SUM_DEBT : column is not present in the DataFrame')

    def _extract_has_credit_loan(self):
        ''' Extract features from the 'CREDIT_TYPE column in the bureau dataframe.
    
            Features Transformed:
            - HAS_CREDIT_LOAN: person have or had the credit loan

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        
        if 'CREDIT_TYPE' in self.df.columns:
            self.df['B_HAS_CREDIT_LOAN'] = np.where(
            self.df['CREDIT_TYPE'] == 'Credit card',
                1,
                0
            )
            filt = self.df['CREDIT_TYPE'].isnull()
            self.df.loc[filt,'B_HAS_CREDIT_LOAN'] = np.nan
            
            features_df = self.df.groupby('SK_ID_CURR')['B_HAS_CREDIT_LOAN'].max().to_frame().reset_index()

            return features_df
        else:
            logger.debug('CREDIT_TYPE : column is not present in the DataFrame')
    def _create_remaining_features(self):
        ''' bootforced features are created to improve the model performance'''
        # Simple aggregation of AMT_CREDIT_SUM
        
        
        # features 1
        total_debt = self.df.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum(min_count=1)
        total_credit = self.df.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum(min_count=1)
        ratio_df = pd.concat([total_debt, total_credit], axis=1).reset_index()
        feature_df_1 = self._create_ratio_feature(ratio_df, 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM', 'B_DEBT_TO_CREDIT_RATIO')
        
        # features 2
        feature_df_2 = self.df.groupby('SK_ID_CURR')['DAYS_CREDIT'].agg(
            B_DAYS_CREDIT_MEAN='mean',
            B_DAYS_CREDIT_MAX='max',
            B_DAYS_CREDIT_MIN='min'
        ).reset_index()
        
        # features 3
        feature_df_3 = self.df.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].agg(
            BUREAU_DAYS_CREDIT_ENDDATE_MEAN='mean',
            BUREAU_DAYS_CREDIT_ENDDATE_MIN='min',
            BUREAU_DAYS_CREDIT_ENDDATE_MAX='max',
        ).reset_index()

        # features 4
        temp = pd.crosstab(self.df['SK_ID_CURR'],self.df['CREDIT_ACTIVE'])

        active = temp.get('Active',0)
        closed = temp.get('Closed',0)

        feature_df_4 = pd.DataFrame({
            'SK_ID_CURR': temp.index,
            'B_ACTIVE_CREDIT_COUNT': active,
            'B_CLOSED_CREDIT_COUNT': closed
        }).reset_index(drop=True)
        
        feature_df_4['B_ACTIVE_CREDIT_RATIO'] = feature_df_4['B_ACTIVE_CREDIT_COUNT'] / feature_df_4[['B_ACTIVE_CREDIT_COUNT','B_CLOSED_CREDIT_COUNT']].sum(axis=1)
        feature_df_4['B_CLOSED_CREDIT_RATIO'] = feature_df_4['B_CLOSED_CREDIT_COUNT'] / feature_df_4[['B_ACTIVE_CREDIT_COUNT','B_CLOSED_CREDIT_COUNT']].sum(axis=1)
        
        # features 5
        self.df['IS_OVERDUE'] = (self.df['CREDIT_DAY_OVERDUE'] > 0).astype(int)
        feature_df_5 = self.df.groupby('SK_ID_CURR')['IS_OVERDUE'].sum().to_frame('B_DEBT_OVERDUE_CNT').reset_index()
        
        # features 6
        feature_df_6 = self.df.groupby('SK_ID_CURR')['IS_OVERDUE'].mean().to_frame('BUREAU_OVERDUE_RATE').reset_index()

        # features 7
        feature_df_7 =  self.df.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count().to_frame('B_CREDIT_COUNT').reset_index()
        
        # features_8
        self.df['B_CREDIT_DURATION'] = self.df['DAYS_CREDIT_ENDDATE'] - self.df['DAYS_CREDIT']
        feature_df_8 = self.df.groupby('SK_ID_CURR')['B_CREDIT_DURATION'].agg(
            B_CREDIT_DURATION_MAX='max',
            B_CREDIT_DURATION_MIN='min',
            B_CREDIT_DURATION_MEAN='mean',
            ).reset_index()
        
        
        feature_df_9 = self.df.groupby('SK_ID_CURR')['AMT_ANNUITY'].agg(
            B_AMT_ANNUITY_MAX='max',
            B_AMT_ANNUITY_MIN='min',
            B_AMT_ANNUITY_MEAN='mean',
            ).reset_index()
        
        #feature_df_10
        self.df['B_IS_BAD'] = self.df['CREDIT_ACTIVE'].isin(['Bad debt','Sold']).astype(int)
        feature_df_10 = self.df.groupby('SK_ID_CURR')['B_IS_BAD'].mean().to_frame('B_BAD_LOAN_RATE').reset_index()
        
        # MERGE CODE
        # Start with first dataframe
        final_features = feature_df_1.copy()

        # List all feature dataframes
        feature_dfs = [
            feature_df_2,
            feature_df_3,
            feature_df_4,
            feature_df_5,
            feature_df_6,
            feature_df_7,
            feature_df_8,
            feature_df_9,
            feature_df_10,
        ]

        # Merge sequentially
        for df in feature_dfs:
            final_features = final_features.merge(
                df,
                on='SK_ID_CURR',
                how='outer'
            )

        return final_features
                        
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
                self._extract_has_credit_loan,
                self._create_remaining_features
                ]

    

            for extractor in self.feature_extractors:
                
                # log the method is running
                method_name = extractor.__name__
                logger.info(f"Current Method Running: {self.__class__.__name__}.{method_name}")        

                
                features_df = extractor()
                
                            
                main_df = main_df.merge(features_df,on='SK_ID_CURR',how='left')
                
                
                new_cols = features_df.columns.drop('SK_ID_CURR')
                main_df[new_cols] = main_df[new_cols].fillna(PLACEHOLDER)


            # ← ADD THESE TWO LINES in every add_features_main loop
            del features_df
            gc.collect()
            logger.info("Aggregated features from the BUREAU dataframe successfully merged into the main dataframe.")

            return main_df

        except Exception as e:
            raise MyException(e,sys,logger)



class InstallmentsPaymentsTransformation(BaseTransformer,RatioFeatureMixin):
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, data_indestion_artifact: DataIngestionArtifact):
        super().__init__(data_transformation_artifact, data_indestion_artifact)

        self.df = self.load_data(
            self.data_indestion_artifact.installment_payments_data,
            )
        

        self.df['PAY_DAYS_DIFF'] = self.df['DAYS_ENTRY_PAYMENT'] - self.df['DAYS_INSTALMENT']

        
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
                filt = (
                    (self.df['DAYS_INSTALMENT'] <= 0) &
                    (self.df['DAYS_INSTALMENT'] >= -frame) &
                    (self.df['NUM_INSTALMENT_VERSION'] != 0)
                )
                
                filt_df = self.df.loc[filt]
                temp = (filt_df.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].max().sub(1)).to_frame()

                temp = temp.groupby('SK_ID_CURR').sum()
                temp = temp.clip(lower=0)
                feature_df = temp.rename(columns={'NUM_INSTALMENT_VERSION':f'IP_NUM_OF_RESHEDULES_{frame}D'})
                if features_df.empty:
                    features_df = feature_df.copy()
                else:
                    features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')

            all_cust_index = self.df['SK_ID_CURR'].sort_values().unique()
            features_df = features_df.reindex(all_cust_index,fill_value=0)
            
            return features_df
           
        else:
            logger.debug('NUM_INSTALMENT_VERSION & DAYS_INSTALMENT : column is not present in the DataFrame')

    def _extract_late_payment_count_ratio_tp(self):
        '''
        Extract late payment count and late payment ratio for recent time periods
        over all loans from the Installments Payments dataframe.

        Late Payment Definition:
            PAY_DAYS_DIFF = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
            Late if PAY_DAYS_DIFF > 0

        Features Transformed
        For each timeframe D in [90, 180, 360, 720, 1080, 2160]:
        - IP_COUNT_LATE_PAYMENTS_{D}D:Number of installments paid late.
        - IP_RATIO_LATE_PAYMENTS_{D}D:Ratio of late payments to total installments in that timeframe.

        Returns:
            features_df(pd.DataFrame):
                DataFrame with SK_ID_CURR index
        '''

        required_cols = [
            'DAYS_ENTRY_PAYMENT',
            'DAYS_INSTALMENT',
            'SK_ID_CURR'
        ]

        if all(col in self.df.columns for col in required_cols):

            # Late payment difference
            self.df['PAY_DAYS_DIFF_TEMP'] = (
                self.df['DAYS_ENTRY_PAYMENT']
                - self.df['DAYS_INSTALMENT']
            )

            time_frames = [90, 180, 360, 720, 1080, 2160]

            features_df = pd.DataFrame()

            for frame in time_frames:

                filt = (
                    (self.df['DAYS_INSTALMENT'] <= 0) &
                    (self.df['DAYS_INSTALMENT'] >= -frame)
                )

                filt_df = self.df.loc[filt]

                # Late payment flag
                filt_df['LATE_PAYMENT_FLAG'] = (filt_df['PAY_DAYS_DIFF_TEMP'] > 0).astype(int)

                # Late payment count
                late_count = (
                    filt_df
                    .groupby('SK_ID_CURR')['LATE_PAYMENT_FLAG']
                    .sum()
                    .to_frame()
                    .rename(
                        columns={
                            'LATE_PAYMENT_FLAG':
                            f'IP_COUNT_LATE_PAYMENTS_{frame}D'
                        })
                    )

                # Total installment count
                total_count = (
                    filt_df
                    .groupby('SK_ID_CURR')['LATE_PAYMENT_FLAG']
                    .count()
                    .to_frame()
                    .rename(
                        columns={
                            'LATE_PAYMENT_FLAG':
                            'TOTAL_INSTALLMENTS'
                        })
                    )

                # Ratio
                ratio_df = (late_count.join(total_count))
                ratio_df = ratio_df.reset_index()
                feature_df = self._create_ratio_feature(ratio_df,f'IP_COUNT_LATE_PAYMENTS_{frame}D','TOTAL_INSTALLMENTS',f'IP_RATIO_LATE_PAYMENTS_{frame}D')
                ratio_df = ratio_df.drop(
                    columns=['TOTAL_INSTALLMENTS']
                )

                if features_df.empty:
                    features_df = feature_df.copy()
                else:
                    features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')

            return features_df

        else:
            logger.debug(
                'Required columns missing for late payment features')

    def _extract_agg_pay_ratio(self):
        ''' extract the pay ratio Avg ,Min,Max PAY_RATIO =  AMT_PAYMENT / AMT_INSTALMENT per customer

            Features Extracted:
            - AVG_PAY_RATIO: average payment ratio per customer
            - MIN_PAY_RATIO: minimum payment ratio per customer
            - MAX_PAY_RATIO: maximum payment ratio per customer

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as
                index and aggregate PAY_RATIO features
                Missing values filled with the placeholder -99999
        '''
        if 'AMT_PAYMENT' in self.df.columns and 'AMT_INSTALMENT' in self.df.columns:
            
            feature_df = self._create_ratio_feature(self.df,'AMT_PAYMENT','AMT_INSTALMENT','IP_RATIO_PAYMENT_INSTALMENT')
          
            
            features_df = feature_df.groupby(by='SK_ID_CURR')['IP_RATIO_PAYMENT_INSTALMENT'].agg(['mean','min','max']).rename(columns={
                "mean":"IP_AVG_RATIO_PAYMENT_INSTALMENT",
                "min":"IP_MIN_RATIO_PAYMENT_INSTALMENT",
                "max":"IP_MAX_RATIO_PAYMENT_INSTALMENT"
            })

            return features_df
            
        else:
            logger.debug('AMT_PAYMENT or DAYS_INSTALMENT : column is not present in the DataFrame')
    
    
    def _extract_early_payments_info(self):
        ''' Extract the early payments ratio and earliest pay flag (3m) from the previous installments dataset
    
            Features Transformed:
            - IP_RATIO_EARLY_PAYMENTS_{frame}D: the ratio of the total payments with payment that is paid before the installment date.
            - IP_RECENT_EARLY_PAYMENT_FLAG_3M : the flag if the person has paid installment early in last 3 months.
            
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
       
        if 'DAYS_ENTRY_PAYMENT' in self.df.columns and 'DAYS_INSTALMENT' in self.df.columns:

            self.df['FLAG_EARLY_PAYMENT'] = (self.df['PAY_DAYS_DIFF'] < 0).astype(int)

            time_frames = [90, 180, 270, 360, 720, 1080, 2160, 2880]
            
            features_df = pd.DataFrame()
            for frame in time_frames:
                filt = (self.df['DAYS_INSTALMENT'] <= 0) & (self.df['DAYS_INSTALMENT'] >= -frame)
                filt_df = self.df.loc[filt]
                
                feature_ratio = filt_df.groupby(by='SK_ID_CURR')['FLAG_EARLY_PAYMENT'].mean().to_frame(f'IP_RATIO_EARLY_PAYMENTS_{frame}D')
                
                if features_df.empty:
                    features_df = feature_ratio.copy()
                else:
                    features_df = features_df.merge(feature_ratio,on='SK_ID_CURR',how='outer')
            
            # recent early payment 3M
            self.df['IP_RECENT_EARLY_PAYMENT_FLAG_3M'] = np.where(
                    ((self.df['FLAG_EARLY_PAYMENT'] == 1) & (self.df['DAYS_INSTALMENT'] >= -90)),1, 0)
            
            features_df_2 = self.df.groupby(by='SK_ID_CURR')['IP_RECENT_EARLY_PAYMENT_FLAG_3M'].max().to_frame()
            features_df = features_df.merge(features_df_2,on='SK_ID_CURR',how='outer')
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
                Missing values filled with the placeholder -66666
        '''

        if 'DAYS_ENTRY_PAYMENT' in self.df.columns and 'DAYS_INSTALMENT' in self.df.columns:
                
            time_frames = [90, 180, 270, 360, 720, 1080, 2160, 2880]
            # empty dataframe to apend the feature into
            

            features_df = pd.DataFrame()
            for frame in time_frames:
                filt  = (self.df['PAY_DAYS_DIFF']>= 0) & (self.df['DAYS_INSTALMENT'] > -frame)
                filt_df_frame = self.df.loc[filt]

                feature_dpd = filt_df_frame.groupby('SK_ID_CURR')['PAY_DAYS_DIFF'].max().to_frame(f'IP_WORST_DPD_{frame}D')
                
                if features_df.empty:
                    features_df = feature_dpd.copy()
                else:
                    features_df = features_df.merge(feature_dpd,on='SK_ID_CURR',how='outer')
                    
            # Is the customer getting worse recently?
            features_df['IP_DPD_TREND'] = features_df['IP_WORST_DPD_90D'] - features_df['IP_WORST_DPD_360D']
            
            features_df = features_df.fillna(DPD_LOAN_DATA_MISSING) #-66666=
            
            return features_df
    
        else:
            logger.debug('DAYS_ENTRY_PAYMENT or DAYS_INSTALMENT : column is not present in the DataFrame')
        

    def _extract_num_underpaid_installments_D(self):
        ''' Create num_underpaid_installment features over multiple time frame
    
            Features Transformed:
            - IP_NUM_UNDERPAID_INSTALLMENTS_ : Number of underpaid installments based on time frame:[180, 360, 720, 1080, 2160, 2880] D

            Returns:
                DataFrame with SK_ID_CURR as index and WORST_DPD_INSTALLMENT_PAYMENTS_XM features
        '''
        
        if 'AMT_INSTALMENT' in self.df.columns and 'AMT_PAYMENT' in self.df.columns:
            
            time_frames = [180, 360, 720, 1080, 2160, 2880]
            features_df = pd.DataFrame()

            for frame in time_frames:
                
                filt  = (self.df['PAY_DAYS_DIFF']>= 0) & (self.df['DAYS_INSTALMENT'] > -frame)

                filt_df = self.df.loc[filt]
                
                filt_df = filt_df.groupby(['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER'])[['AMT_INSTALMENT','AMT_PAYMENT']].sum()

                filt_df['NUM_UNDERPAID_INSTALLMENTS'] = (filt_df['AMT_INSTALMENT'] > filt_df['AMT_PAYMENT']).astype('int')

                num_underpaid_installments = filt_df.groupby('SK_ID_CURR')['NUM_UNDERPAID_INSTALLMENTS'].sum().to_frame(f'IP_NUM_UNDERPAID_INSTALLMENTS_{frame}D')

                if features_df.empty:
                    features_df = num_underpaid_installments.copy()
                else:
                    features_df = features_df.merge(num_underpaid_installments,on='SK_ID_CURR',how='outer')
                        
            all_cust_index = self.df['SK_ID_CURR'].unique()
            features_df = features_df.reindex(all_cust_index, fill_value=0)            
            return features_df
        
        else:
            logger.debug('AMT_INSTALMENT or AMT_PAYMENT : column is not present in the DataFrame')


    def _extract_agg_pay_diff(self):
        ''' aggregate min,max,sum,mean  of the PAY_DIFF and extract them
    
            Features Transformed:
            - "IP_MIN_RATIO_INSTALMENT_PAYMENT": min of the total pay_diff
            - "IP_MAX_RATIO_INSTALMENT_PAYMENT": sum of the total pay_diff
            - "IP_SUM_RATIO_INSTALMENT_PAYMENT": max of the total pay_diff
            - "IP_MEAN_RATIO_INSTALMENT_PAYMENT": mean of the total pay_diff
            
            Returns:
                DataFrame with SK_ID_CURR as index and aggregated features of the payment difference
        '''
        
        if 'AMT_INSTALMENT' in self.df.columns and 'AMT_PAYMENT' in self.df.columns:
            
            filt_df = self.df.groupby(['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER'])[['AMT_INSTALMENT','AMT_PAYMENT']].sum()

            filt_df['PAY_DIFF'] = filt_df['AMT_INSTALMENT'] - filt_df['AMT_PAYMENT']

            features_df =  filt_df.groupby(by='SK_ID_CURR')['PAY_DIFF'].agg(['mean','min','max','sum']).rename(columns= {
                "mean":"IP_MEAN_INSTALMENT_PAYMENT_DIFF",
                "min":"IP_MIN_INSTALMENT_PAYMENT_DIFF",
                "max":"IP_MAX_INSTALMENT_PAYMENT_DIFF",
                "sum":"IP_SUM_INSTALMENT_PAYMENT_DIFF"
                })

            return features_df
        
        else:
            logger.debug('AMT_INSTALMENT or AMT_PAYMENT : column is not present in the DataFrame')

    def _extract_missed_payment_count_ratio_tp(self):
        '''
        Extract missed payment count and missed payment ratio
        for recent time periods from Installments Payments dataframe.

        Missed Payment Definition:
            AMT_PAYMENT == 0
            OR AMT_PAYMENT is NaN
            OR DAYS_ENTRY_PAYMENT is NaN

        Features Transformed:
        For each timeframe D in [90, 180, 360, 720, 1080, 2160]:
        - IP_COUNT_MISSED_PAYMENTS_{D}D : Number of missed payments.
        - IP_RATIO_MISSED_PAYMENTS_{D}D :atio of missed payments to total installments.

        Returns:
            features_df(pd.DataFrame):
                DataFrame with SK_ID_CURR index
        '''

        required_cols = [
            'AMT_PAYMENT',
            'DAYS_ENTRY_PAYMENT',
            'DAYS_INSTALMENT',
            'SK_ID_CURR'
        ]

        if all(col in self.df.columns for col in required_cols):


            time_frames = [90, 180, 360, 720, 1080, 2160]

            features_df = pd.DataFrame()

            for frame in time_frames:

                filt = (
                    (self.df['DAYS_INSTALMENT'] <= 0) &
                    (self.df['DAYS_INSTALMENT'] >= -frame)
                )

                filt_df = self.df.loc[filt]

                # Missed payment flag
                filt_df['MISSED_PAYMENT_FLAG'] = (
                    (filt_df['AMT_PAYMENT'] == 0) |
                    (filt_df['AMT_PAYMENT'].isna()) |
                    (filt_df['DAYS_ENTRY_PAYMENT'].isna())
                ).astype(int)

                # Missed count
                missed_count = (
                    filt_df
                    .groupby('SK_ID_CURR')['MISSED_PAYMENT_FLAG']
                    .sum()
                    .to_frame()
                    .rename(columns={
                        'MISSED_PAYMENT_FLAG':
                        f'IP_COUNT_MISSED_PAYMENTS_{frame}D'
                    })
                )

                # Total installments
                total_count = (
                    filt_df
                    .groupby('SK_ID_CURR')['MISSED_PAYMENT_FLAG']
                    .count()
                    .to_frame()
                    .rename(columns={
                        'MISSED_PAYMENT_FLAG':
                        'TOTAL_INSTALLMENTS'
                    })
                )

                ratio_df = missed_count.join(total_count)
                ratio_df = ratio_df.reset_index()
                # Create ratio safely
                feature_df = self._create_ratio_feature(
                    ratio_df,
                    f'IP_COUNT_MISSED_PAYMENTS_{frame}D',
                    'TOTAL_INSTALLMENTS',
                    f'IP_RATIO_MISSED_PAYMENTS_{frame}D'
                )

                if features_df.empty:
                    features_df = feature_df.copy()
                else:
                    features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')
           
            # Ensure all customers included
            all_cust_index = (
                self.df['SK_ID_CURR']
                .sort_values()
                .unique()
            )

            features_df = features_df.reindex(
                all_cust_index,
                fill_value=0
            )
            return features_df

        else:
            logger.debug(
                'Required columns missing for missed payment features'
            )

    def _extract_dpd_severity_counts(self):
        '''
        Extract DPD (Days Past Due) severity count features
        from Installments Payments dataframe.

        IP_COUNT_DPD_GT0   → Late payments (>0 days)
        IP_COUNT_DPD_GT30  → 30+ days late
        IP_COUNT_DPD_GT60  → 60+ days late
        IP_COUNT_DPD_GT90  → 90+ days late
        '''

        required_cols = [
            'SK_ID_CURR',
            'DAYS_ENTRY_PAYMENT',
            'DAYS_INSTALMENT'
        ]

        # Validate columns
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(
                "Missing required columns for DPD feature extraction"
            )


        # Create PAY_DAYS_DIFF
        self.df['PAY_DAYS_DIFF'] = (
            self.df['DAYS_ENTRY_PAYMENT']
            - self.df['DAYS_INSTALMENT']
        )

        # Create severity flags
        self.df['DPD_GT0'] = (self.df['PAY_DAYS_DIFF'] > 0).astype(int)
        self.df['DPD_GT30'] = (self.df['PAY_DAYS_DIFF'] > 30).astype(int)
        self.df['DPD_GT60'] = (self.df['PAY_DAYS_DIFF'] > 60).astype(int)
        self.df['DPD_GT90'] = (self.df['PAY_DAYS_DIFF'] > 90).astype(int)

        # Aggregate counts per customer
        features_df = (
            self.df
            .groupby('SK_ID_CURR')[
                ['DPD_GT0','DPD_GT30','DPD_GT60','DPD_GT90']
            ]
            .sum()
        )

        # Rename columns
        features_df = features_df.rename(columns={
            'DPD_GT0': 'IP_COUNT_DPD_GT0',
            'DPD_GT30': 'IP_COUNT_DPD_GT30',
            'DPD_GT60': 'IP_COUNT_DPD_GT60',
            'DPD_GT90': 'IP_COUNT_DPD_GT90'
        })

        return features_df.reset_index()
    import pandas as pd

    def _extract_installment_volume_features(self):
        """
        Basic installment volume features.
        Creates:
            IP_TOTAL_INSTALLMENTS_COUNT  -> total installment records
            IP_NUM_PREV_LOANS            -> number of unique previous loan
        """

        # Group and aggregate
        features_df = (
            self.df
            .groupby('SK_ID_CURR')
            .agg(
                IP_TOTAL_INSTALLMENTS_COUNT=('SK_ID_CURR', 'size'),
                IP_NUM_PREV_LOANS=('SK_ID_PREV', 'nunique')
            )
        )

        return features_df.reset_index()

    def _extract_completed_loans_feature(self):
        """
        Count number of fully completed previous loans.

        Creates:
            IP_NUM_COMPLETED_LOANS: Loan is completed if ALL installments
            under that SK_ID_PREV were paid.

        """

        # Flag paid installment
        self.df['PAID_FLAG'] = (
            self.df['AMT_PAYMENT'] >= self.df['AMT_INSTALMENT']
        ).astype(int)

        # Check if each loan is fully paid
        loan_status = (
            self.df
            .groupby(['SK_ID_CURR', 'SK_ID_PREV'])['PAID_FLAG']
            .min()   # if any unpaid → becomes 0
            .reset_index()
        )

        # Count completed loans
        features_df = (
            loan_status
            .groupby('SK_ID_CURR')['PAID_FLAG']
            .sum()
            .to_frame('IP_NUM_COMPLETED_LOANS')
        )

        return features_df.reset_index()

    def _extract_payment_coverage_features(self):
        """
        Payment coverage features per timeframe.
        Creates (for each timeframe D in [360, 720, 1080]):

            IP_SUM_AMT_PAYMENT_{D}D
            IP_SUM_AMT_INSTALMENT_{D}D
            IP_RATIO_AMT_PAID_OWED_{D}D

            Sum of actual payments vs sum of owed installments
            within recent time windows.

        """

        required_cols = [
            'SK_ID_CURR',
            'AMT_PAYMENT',
            'AMT_INSTALMENT',
            'DAYS_INSTALMENT'
        ]

        if not all(col in self.df.columns for col in required_cols):
            raise ValueError("Missing required columns")


        time_frames = [360, 720, 1080]

        features_df = pd.DataFrame()

        for frame in time_frames:

            # Filter timeframe
            filt = (
                (self.df['DAYS_INSTALMENT'] <= 0) &
                (self.df['DAYS_INSTALMENT'] >= -frame)
            )

            filt_df = self.df.loc[filt]

            # Aggregate sums
            temp = (
                filt_df
                .groupby('SK_ID_CURR')
                .agg(
                    **{
                        f'IP_SUM_AMT_PAYMENT_{frame}D':
                            ('AMT_PAYMENT', 'sum'),

                        f'IP_SUM_AMT_INSTALMENT_{frame}D':
                            ('AMT_INSTALMENT', 'sum')
                    }
                )
            )
            temp = temp.reset_index()

            # Create ratio safely
            paid_col = f'IP_SUM_AMT_PAYMENT_{frame}D'
            owed_col = f'IP_SUM_AMT_INSTALMENT_{frame}D'

            ratio_col = f'IP_RATIO_AMT_PAID_OWED_{frame}D'
            
            
            # Use your ratio function
            ratio_df = self._create_ratio_feature(
                temp,
                paid_col,
                owed_col,
                ratio_col
            )

            # Merge
            if features_df.empty:
                features_df = ratio_df.copy()
            else:
                features_df = features_df.merge(ratio_df,on='SK_ID_CURR',how='outer')
        return features_df
    
    def add_features_main(self,main_df):
        '''Extract and Create feature from installments payment dataset and append in main application  dataframe'''   
        try:
            self.feature_extractors  = [
                self._extract_number_of_reshedules_tp,
                self._extract_agg_pay_ratio,
                self._extract_early_payments_info,
                self._extract_worst_dpd_features_installmentsDF,
                self._extract_num_underpaid_installments_D,
                self._extract_agg_pay_diff,
                self._extract_late_payment_count_ratio_tp,
                self._extract_missed_payment_count_ratio_tp,
                self._extract_dpd_severity_counts,
                self._extract_installment_volume_features,
                self._extract_completed_loans_feature,
                self._extract_payment_coverage_features
                
            ]

            for extractor in self.feature_extractors:
                 # log the method is running
                method_name = extractor.__name__
                logger.info(f"Current Method Running: {self.__class__.__name__}.{method_name}")        


                features_df = extractor()

                features_cols = features_df.columns.to_list()

                main_df = main_df.merge(features_df,on='SK_ID_CURR',how='left')
                main_df[features_cols] = main_df[features_cols].fillna(PLACEHOLDER)

            del features_df
            gc.collect()        
            logger.info("Aggregated features from the installlments payments dataframe successfully merged into the Previous Application dataframe.")

            return main_df
         
        except Exception as e:
            raise MyException(e,sys,logger)
 


class PosCashBalanceTransformation(BaseTransformer,RatioFeatureMixin):
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, data_indestion_artifact: DataIngestionArtifact):
        super().__init__(data_transformation_artifact, data_indestion_artifact)

        self.df = self.load_data(
            self.data_indestion_artifact.pos_cash_data,

        )


        self.all_cust_index = self.df['SK_ID_CURR'].unique()



    def _extract_has_risky_contract_status(self):
        ''' Extract features from the 'NAME_CONTRACT_STATUS  from the pos_cash_balance dataset.
    
            Features Transformed:
            - PCB_FLAG_RISKY_CONTRACT_STATUS_{XM}: Flag if the customer had any risky contract status in the last X months. NAME_CONTRACT_STATUS 'Demand','Amortized debt','Returned to the store'.
            - PCB_RATIO_RISKY_STATUS_{XM}: Ratio of risky contract statuses over total records in the last X months.
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
            '''
        
        if 'NAME_CONTRACT_STATUS' in self.df.columns :
            time_frames = [3,6,9,12,24] # 3,6,9,12,24 Months
            risky_categories = ['Demand','Amortized debt','Returned to the store']
            
            self.df['PCB_FLAG_RISKY_CONTRACT_STATUS'] = np.where(self.df['NAME_CONTRACT_STATUS'].isin(risky_categories),
                    1,
                    0)
            
            features_df = pd.DataFrame()
            for frame in time_frames:
                filt_df = self.df[(self.df['MONTHS_BALANCE'] >= -frame) & (self.df['MONTHS_BALANCE'] <= 0)]
                
                agg_df = filt_df.groupby('SK_ID_CURR').agg(
                    PCB_FLAG_RISKY_CONTRACT_STATUS=('PCB_FLAG_RISKY_CONTRACT_STATUS','max'),
                    PCB_RATIO_RISKY_STATUS= ('PCB_FLAG_RISKY_CONTRACT_STATUS','mean')
                    )
                agg_df = agg_df.rename(columns={'PCB_FLAG_RISKY_CONTRACT_STATUS':f'PCB_FLAG_RISKY_CONTRACT_STATUS_{frame}M','PCB_RATIO_RISKY_STATUS':f'PCB_RATIO_RISKY_STATUS_{frame}M'})
                agg_df  = agg_df.reset_index()
                if features_df.empty:
                    features_df = agg_df
                else:
                    features_df = features_df.merge(agg_df,on='SK_ID_CURR',how='outer')
                
            filt = self.df['PCB_FLAG_RISKY_CONTRACT_STATUS'] ==1
            temp = self.df.loc[filt]
            feature = temp.groupby(by='SK_ID_CURR')['MONTHS_BALANCE'].max().to_frame('PCB_LATEST_RISKY_STATUS_FLAG').reset_index()    
            features_df = features_df.merge(feature,on='SK_ID_CURR',how='outer')
            return features_df
        
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
            time_frames = [3,6,9,12,24]
            features_df = pd.DataFrame()

            for frame in time_frames:

                filt_df = self.df[(self.df['MONTHS_BALANCE'] >= -frame) & 
                                (self.df['MONTHS_BALANCE'] <= 0)]  
                            
                # vectorized flags
                filt_df['IS_ACTIVE'] = filt_df['NAME_CONTRACT_STATUS'].eq('Active')
                filt_df['IS_COMPLETED'] = filt_df['NAME_CONTRACT_STATUS'].eq('Completed')

                loan_status = filt_df.groupby(['SK_ID_CURR','SK_ID_PREV'])[['IS_ACTIVE','IS_COMPLETED']].max()

                # Active AND not Completed loans
                valid_loans = loan_status[(loan_status['IS_ACTIVE']) & (~loan_status['IS_COMPLETED'])]
                valid_loans = valid_loans.reset_index()
                feature_df = valid_loans.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame(f'PCB_NUM_ACTIVE_LOANS_{frame}M')

                if features_df.empty:
                    features_df = feature_df
                else:
                    features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')
                    
            features_df['PCB_ACTIVE_LOAN_TREND_3M_12M'] = features_df['PCB_NUM_ACTIVE_LOANS_3M'] - features_df['PCB_NUM_ACTIVE_LOANS_12M']
            features_df = features_df.reindex(self.all_cust_index,fill_value=0) 
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
            feature_df = filt_df.groupby(by='SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].agg(
                PCB_MAX_REMAINING_INSTALLMENTS = 'max',
                PCB_SUM_REMAINING_INSTALLMENTS = 'sum',
                PCB_MIN_REMAINING_INSTALLMENTS = 'min',
                
            ).reset_index()
                        
            filt_df['PCB_INSTALMENT_PROGRESS'] = (filt_df['CNT_INSTALMENT'] - filt_df['CNT_INSTALMENT_FUTURE']) / filt_df['CNT_INSTALMENT']
            feature_df1 = filt_df.groupby('SK_ID_CURR')['PCB_INSTALMENT_PROGRESS'].agg(
                PCB_INSTALMENT_PROGRESS_MEAN ='mean',
                PCB_INSTALMENT_PROGRESS_MAX ='max',    
            )            
            
            feature_df = feature_df.reindex(self.all_cust_index,fill_value=0)
            feature_df = feature_df.merge(feature_df1,on='SK_ID_CURR',how='outer')

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
                Missing values filled with the placeholder -66666

        '''
        if 'SK_DPD' in self.df.columns and 'MONTHS_BALANCE' in self.df.columns:
            time_frames = [3, 6, 9, 12, 24, 36, 72, 96]
            # empty dataframe to apend the feature into
            features_df = pd.DataFrame()

            for frame in time_frames:
                filt = (self.df['MONTHS_BALANCE'] >= -frame) & (self.df['MONTHS_BALANCE'] <= 0)
                temp = self.df.loc[filt]
                feature_dpd = temp.groupby(by='SK_ID_CURR')['SK_DPD'].max().to_frame(f'PCB_WORST_DPD_POS_CASH_{frame}M').reset_index()
                
                if features_df.empty:
                    features_df = feature_dpd
                else:
                    features_df = features_df.merge(feature_dpd,on='SK_ID_CURR',how='outer')

            features_df = features_df.fillna(DPD_LOAN_DATA_MISSING) # -66666
                
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
                filt = (self.df['MONTHS_BALANCE'] >= -frame) & (self.df['MONTHS_BALANCE'] <= 0)
                temp = self.df.loc[filt]
                feature_dpd = temp.groupby(by='SK_ID_CURR')['SK_DPD_DEF'].max().to_frame(f'WORST_DPD_DEF_POS_CASH_{frame}M').reset_index()
                
                
                if features_df.empty:
                    features_df = feature_dpd
                else:
                    features_df = features_df.merge(feature_dpd,on='SK_ID_CURR',how='outer')

            features_df = features_df.fillna(DPD_LOAN_DATA_MISSING) # -66666
            return features_df
        else:
            logger.debug('STATUS and SK_DPD_DEF: column is not present in the DataFrame')
                  
    def _extract_month_cnt_history(self):
        ''' how many months history they have in data'''
        
        pcb_history_length = self.df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].min().abs()
        feature_df = pcb_history_length.to_frame('PCB_HISTORY_LENGTH').reset_index()
        
        return feature_df
    def _extract_loan_completion_ratio(self):
        ''' loan completed to total ratio'''
        
        PCB_TOTAL_PREV_LOANS = self.df.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame('PCB_TOTAL_PREV_LOANS').reset_index()
        temp = self.df[self.df['NAME_CONTRACT_STATUS'].eq('Completed')]
        PCB_NUM_COMPLETED_LOANS  = temp.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame('PCB_NUM_COMPLETED_LOANS').reset_index()
        feature_df = PCB_TOTAL_PREV_LOANS.merge(PCB_NUM_COMPLETED_LOANS,on='SK_ID_CURR',how='outer')
        feature = self._create_ratio_feature(feature_df,'PCB_NUM_COMPLETED_LOANS','PCB_TOTAL_PREV_LOANS','PCB_LOAN_COMPLETION_RATE')
        return feature
    
    def _extract_dpd_buckets_data(self):
        '''# Count of payments late by severity bucket
        # SK_DPD > 0
        IP_COUNT_DPD_GT30: SK_DPD > 30
        IP_COUNT_DPD_GT60: SK_DPD > 60
        IP_COUNT_DPD_GT90: SK_DPD > 90
        '''
        
        self.df['DPD_GT0'] = (self.df['SK_DPD'] > 0).astype(int)
        self.df['DPD_GT30'] = (self.df['SK_DPD'] > 30).astype(int)
        self.df['DPD_GT60'] = (self.df['SK_DPD'] > 60).astype(int)
        self.df['DPD_GT90'] = (self.df['SK_DPD'] > 90).astype(int)

        features_df = (
            self.df
            .groupby('SK_ID_CURR')[['DPD_GT0','DPD_GT30','DPD_GT60','DPD_GT90']]
            .sum()
        )

        # Rename columns
        features_df = features_df.rename(columns={
            'DPD_GT0': 'PCB_COUNT_DPD_GT0',
            'DPD_GT30': 'PCB_COUNT_DPD_GT30',
            'DPD_GT60': 'PCB_COUNT_DPD_GT60',
            'DPD_GT90': 'PCB_COUNT_DPD_GT90'
        })

        features_df.reset_index()
        return features_df
                
    def add_features_main(self,main_df):

        '''Extract and Create feature Pos Cash Balance and append in Main Dataframe'''   
        try:
            self.feature_extractors  = [
                self._extract_has_risky_contract_status,
                self._extract_num_active_loans_XM,
                self._extract_cnt_installment_future,
                self._extract_worst_dpd_features_pos_cash,
                self._extract_worst_dpd_def_features_pos_cash,
                self._extract_loan_completion_ratio,
                self._extract_month_cnt_history
            ]

           
            for extractor in self.feature_extractors:
                
                # log the method is running
                method_name = extractor.__name__
                logger.info(f"Current Method Running: {self.__class__.__name__}.{method_name}")        

                
                features_df = extractor()
                
                       
                features_cols = features_df.columns.to_list()
                
                
                main_df = main_df.merge(features_df,on='SK_ID_CURR',how='left')
                main_df[features_cols] = main_df[features_cols].fillna(PLACEHOLDER)
                
            del features_df
            gc.collect()   
            logger.info("Aggregated features from the pos cash balance dataframe successfully merged into the main  Application dataframe.")

            return main_df
         
        except Exception as e:
            raise MyException(e,sys,logger)


class PreviousApplicationsTransformation(BaseTransformer,RatioFeatureMixin):

    def __init__(self, data_transformation_artifact: DataTransformationArtifact, data_indestion_artifact: DataIngestionArtifact):
        super().__init__(data_transformation_artifact, data_indestion_artifact)

        self.df = self.load_data(
            self.data_indestion_artifact.previous_application_data,
        )
       


    
    def _extract_flag_has_credit_history(self):

        ''' flag the client if the client have any credit card history from the previous application dataset

            Features Transformed:
            - FLAG_HAS_CREDIT_CARD_HISTORY : flag 1 if person own the credit card  history else 0 

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'NAME_PORTFOLIO'}
        if required_cols.issubset(self.df.columns):

            self.df['PA_FLAG_HAS_CREDIT_CARD_HISTORY'] = (self.df['NAME_PORTFOLIO'] =='Credit').astype(int)
            feature_df = self.df.groupby('SK_ID_CURR')['PA_FLAG_HAS_CREDIT_CARD_HISTORY'].max().to_frame()
            
            return feature_df

        else:
            logger.debug(" NAME_PORTFOLIO column are not present in the DataFrame")   

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

                filt = (self.df['NAME_PORTFOLIO'] == category)
                filt_df = self.df.loc[filt]

                agg_df = filt_df.groupby(by='SK_ID_CURR').agg(
                    AVG_AMT_ANNUITY = ('AMT_ANNUITY', 'mean'),
                    SUM_AMT_CREDIT=('AMT_CREDIT', 'sum'),
                    SUM_AMT_ANNUITY=('AMT_ANNUITY', 'sum')
                    ).reset_index()

                ## ratio 

                fe = self._create_ratio_feature(agg_df,'SUM_AMT_CREDIT','SUM_AMT_ANNUITY','AMT_CREDIT_TO_ANNUITY_RATIO')
                
                agg_df = agg_df.merge(fe, on='SK_ID_CURR', how='left')            
                
                
                agg_df.rename(columns={
                    'AMT_CREDIT_TO_ANNUITY_RATIO':f'PA_RATIO_AMT_CREDIT_TO_ANNUITY_{category.upper()}',
                    'AVG_AMT_ANNUITY':f'PA_AVG_AMT_ANNUITY_{category.upper()}'
                },inplace=True)
                
                feature_df = agg_df[['SK_ID_CURR',f'PA_RATIO_AMT_CREDIT_TO_ANNUITY_{category.upper()}',f'PA_AVG_AMT_ANNUITY_{category.upper()}']]

                if features_df.empty:
                    features_df = feature_df.copy()
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
                    filt = ((self.df['NAME_PORTFOLIO'] == category) & (self.df['DAYS_DECISION'] <= 0) &(self.df['DAYS_DECISION'] > -frame))
                    filt_df = self.df.loc[filt]

                    feature_df = filt_df.groupby(by='SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame(f'PA_NUM_LOANS_{category}_{frame}D')
                    
                    if features_df.empty:
                        features_df = feature_df.copy()
                    else:
                        features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')
                    
            features_df = features_df.reset_index()
            # filling 0 because basically 0 loans
            return features_df.fillna(0)
        
        else:
            logger.debug('DAYS_DECISION and NAME_PORTFOLIO : column are not present in the DataFrame')

    def _extract_avg_credit_client(self):
        ''' Extract features from the 'AMT_CREDIT' in the previous application dataset.

            Features Transformed:
            - PA_AVG_AMT_CREDIT:  average credit amount allocated  per client 
            - PA_MAX_AMT_CREDIT:  max credit amount allocated  per client 
            - PA_MIN_AMT_CREDIT:  min credit amount allocated  per client 
            - PA_SUM_AMT_CREDIT:  sum credit amount allocated  per client 

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'AMT_CREDIT' in self.df.columns :
            # AMT_CREDIT means the actual amount of loan the client gets approved
            feature_df = self.df.groupby(by='SK_ID_CURR')['AMT_CREDIT'].agg(
                PA_AVG_AMT_CREDIT='mean',
                PA_MAX_AMT_CREDIT='max',
                PA_MIN_AMT_CREDIT='min',
                PA_SUM_AMT_CREDIT='sum',
                
            ).reset_index()
        
            
            return feature_df
        
        else:
            logger.debug('AMT_CREDIT : column is not present in the DataFrame')


    def _extract_avg_credit_application_ratio(self):
        ''' Extract ratio of the credit / application for different categories such as pos,cash,credit
            from previous application dataset

            Features Transformed:
            - PA_RATIO_CREDIT_APPLICATION{X}:  AMT_CREDIT / AMT_APPLICATION and the average of this value per customer
                X is categories
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'AMT_CREDIT' in self.df.columns and 'AMT_APPLICATION' in self.df.columns  :
            category_types = ['POS', 'Cash', 'Cards']
            features_df = pd.DataFrame()

            for category in category_types:
                filt = (self.df['NAME_PORTFOLIO'] == category)
                filt_df = self.df.loc[filt]

                agg_df = filt_df.groupby(by='SK_ID_CURR').agg(
                    SUM_AMT_CREDIT = ('AMT_CREDIT','sum'),
                    SUM_AMT_APPLICATION= ('AMT_APPLICATION','sum')
                )
                ## ratio 
                agg_df[f'PA_RATIO_CREDIT_APPLICATION_{category}'] = np.select(
                condlist=[
                    agg_df['SUM_AMT_CREDIT'].isna() & agg_df['SUM_AMT_APPLICATION'].isna(),
                    agg_df['SUM_AMT_APPLICATION'].isna(),
                    agg_df['SUM_AMT_CREDIT'].isna()
                ],
                choicelist=[
                    DN_MISSING_PLACEHOLDER,
                    D_MISSING_PLACEHOLDER,
                    N_MISSING_PLACEHOLDER
                ],
                default = agg_df['SUM_AMT_CREDIT'] / agg_df['SUM_AMT_APPLICATION'] 
                )
                
                feature_df = agg_df[[f'PA_RATIO_CREDIT_APPLICATION_{category}']]
                
                if features_df.empty:
                    features_df = feature_df.copy()
                else:
                    features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')
                
                
            return features_df.reset_index()
        else:
            logger.debug('AMT_CREDIT and  AMT_APPLICATION : column are not present in the DataFrame')

   

    def _extract_avg_down_payment_rate(self):
        ''' Extract features from the 'RATE_DOWN_PAYMENT' in the previous application dataset.
            downpayment is done for the POS category only

            Features Transformed:
            - PA_AVG_DOWN_PAYMENT_RATE:  avg downpayment rate for the per client 
            - PA_MIN_DOWN_PAYMENT_RATE:  min downpayment rate for the per client 
            - PA_MAX_DOWN_PAYMENT_RATE:  max downpayment rate for the per client 
            - PA_STD_DOWN_PAYMENT_RATE:  std downpayment for the per client 
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'RATE_DOWN_PAYMENT' in self.df.columns and 'NAME_PORTFOLIO' in self.df.columns:

            filt_df = self.df[self.df['NAME_PORTFOLIO'] =='POS']

            feature_df =  filt_df.groupby(by='SK_ID_CURR')['RATE_DOWN_PAYMENT'].agg(
                PA_AVG_DOWN_PAYMENT_RATE='mean',
                PA_MIN_DOWN_PAYMENT_RATE='min',
                PA_MAX_DOWN_PAYMENT_RATE='max',
                PA_STD_DOWN_PAYMENT_RATE='std'
                ).reset_index()
            

            
            return feature_df
        
        else:
            logger.debug('RATE_DOWN_PAYMENT and NAME_PORTFOLIO : column are not present in the DataFrame')
            

    def _extract_avg_goods_price(self):
        ''' Extract features from the 'AMT_GOODS_PRICE' in the previous application dataset.

            Features Transformed:
            - PA_AVG_AMT_GOODS_PRICE: avg amount of prvious goods that the loan is applied 
            - PA_MIN_AMT_GOODS_PRICE: Min amount of prvious goods that the loan is applied 
            - PA_MAX_AMT_GOODS_PRICE: Max amount of prvious goods that the loan is applied 
            - PA_STD_AMT_GOODS_PRICE: std amount of prvious goods that the loan is applied 

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'AMT_GOODS_PRICE' in self.df.columns :
            
            feature_df =  self.df.groupby(by='SK_ID_CURR')['AMT_GOODS_PRICE'].agg(
                PA_AVG_AMT_GOODS_PRICE='mean',
                PA_MIN_AMT_GOODS_PRICE='min',
                PA_MAX_AMT_GOODS_PRICE='max',
                PA_STD_AMT_GOODS_PRICE='std'
                ).reset_index()

            return feature_df
        
        else:
            logger.debug('AMT_GOODS_PRICE  : column are not present in the DataFrame')

    
    def _extract_mean_privileged_interest_rate(self):
        ''' Extract features from the 'RATE_INTEREST_PRIVILEGED' in the previous application dataset.

            Features Transformed:
            
            - AVG_PRIVILEGED_RATE_FLAG:  proportion of applications where the loan interest rate was privileged
            - FLAG_HAD_RATE_INTEREST_PRIVILEGED: flag if client ever had the interest rate was privileged

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'RATE_INTEREST_PRIVILEGED' in self.df.columns :

            self.df['FLAG_HAD_RATE_INTEREST_PRIVILEGED'] = (self.df['RATE_INTEREST_PRIVILEGED'] > 0).astype(int)

            feature_df_1 = self.df.groupby(by='SK_ID_CURR')['FLAG_HAD_RATE_INTEREST_PRIVILEGED'].mean().to_frame('PA_AVG_PRIVILEGED_RATE_FLAG')
            feature_df_2 = self.df.groupby(by='SK_ID_CURR')['FLAG_HAD_RATE_INTEREST_PRIVILEGED'].max().to_frame('FLAG_HAD_RATE_INTEREST_PRIVILEGED')
            feature_df_1 = feature_df_1.merge(feature_df_2,on='SK_ID_CURR',how='outer')
            
            return feature_df_1
        
        else:
            logger.debug('RATE_INTEREST_PRIVILEGED  : column are not present in the DataFrame')


    def _extract_approved_refused_loan_ratios(self):
        ''' Extract  
                approved / total loan ratio ,
                refused / total loan ratio 
                from the previous application dataset.

            Features Transformed:
            - PA_RATIO_REFUSED_LOANS: ratio of the refused loans.  refused / approved + refused
            - PA_RATIO_APPROVED_LOANS: ratio of the refused loans.  approved / approved + refused

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        if 'NAME_CONTRACT_STATUS' in self.df.columns:
            
            features_df = pd.DataFrame()
            temp = pd.crosstab(self.df['SK_ID_CURR'],self.df['NAME_CONTRACT_STATUS'],dropna=False)
            temp['TOTAL_LOANS'] = temp.get('Approved', 0) + temp.get('Refused', 0)
            temp = temp.reset_index()
            # to avoid Division By 0 Error
            filt  =temp['TOTAL_LOANS'] > 0
            filt_df = temp.loc[filt]
            
            features_1 = self._create_ratio_feature(filt_df,'Refused','TOTAL_LOANS','PA_RATIO_REFUSED_LOANS')
            features_2 = self._create_ratio_feature(filt_df,'Approved','TOTAL_LOANS','PA_RATIO_APPROVED_LOANS')
            
            features_df = features_1.merge(features_2,on='SK_ID_CURR',how='outer')

            return features_df
        else:
            logger.debug('NAME_CONTRACT_STATUS  : column are not present in the DataFrame')


    def _extract_num_refused_loans_d(self):

        ''' Extract features from the 'DAYS_DECISION' and  NAME_CONTRACT_STATUS in the previous application dataset.

            Features Transformed:
            - LOANS_REFUSED_RECENT_{XD}: num of refused loans in the last days_window
                time_frames = [90,180,360, 720, 1080, 1440] 
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        
        if 'DAYS_DECISION' in self.df.columns and 'NAME_CONTRACT_STATUS' in self.df.columns :
            time_frames = [90,180,360, 720, 1080, 1440] 
            features_df = pd.DataFrame()

            for frame in time_frames:
                    
                filt = ((self.df['DAYS_DECISION'] > -frame) & (self.df['NAME_CONTRACT_STATUS'] == 'Refused'))

                temp = self.df.loc[filt]

                feature_df = temp.groupby(by='SK_ID_CURR')['SK_ID_PREV'].nunique().to_frame(f'PA_LOANS_REFUSED_RECENT_{frame}D')
                if features_df.empty:
                    features_df = feature_df
                else:
                    features_df =features_df.merge(feature_df,on='SK_ID_CURR',how='outer')
                
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
            feature_df = self.df.groupby('SK_ID_CURR')['DAYS_DECISION'].max().to_frame('PA_DAYS_SINCE_LAST_LOAN_APPLY').reset_index()

            feature_df['PA_DAYS_SINCE_LAST_LOAN_APPLY'] = - feature_df['PA_DAYS_SINCE_LAST_LOAN_APPLY']

            return feature_df
        else:
            logger.debug('DAYS_DECISION column not present in the DataFrame')

    def _extract_num_hc_reject_loans_d(self):

        ''' extract the number of loans rejected due to HC reject reason per client  in time frames

            Features Transformed:
            - NUM_HC_REJECT_REASON_XD: Flag if the client has the loan  rejected loan due to the HC reject reason in last 2 year
            - FLAG_HC_REJECT_REASON: flag the client if he ever had the rejected loan due to hc reject reason
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'CODE_REJECT_REASON' in self.df.columns and 'DAYS_DECISION' in self.df.columns:
            time_frames =[90, 180, 270, 360, 720,1080]
            features_df = pd.DataFrame()
            self.df['FLAG_HC_REJECT_REASON'] = (self.df['CODE_REJECT_REASON'] =='HC').astype(int)

            for frame in time_frames:    
                filt = (self.df['DAYS_DECISION'] > -frame)
                temp = (self.df.loc[filt])

                feature_df = temp.groupby(by='SK_ID_CURR')['FLAG_HC_REJECT_REASON'].sum().to_frame(f'PA_NUM_HC_REJECT_REASON_{frame}D').reset_index()
                
                if features_df.empty:
                    features_df = feature_df
                else:
                    features_df = features_df.merge(feature_df, on='SK_ID_CURR', how='outer')

            flag_hc=  self.df.groupby(by='SK_ID_CURR')['FLAG_HC_REJECT_REASON'].max().to_frame().reset_index()
            features_df = features_df.merge(flag_hc,on='SK_ID_CURR',how='outer')
            

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
            
            self.df['NUM_HC_REJECT_REASON'] = (self.df['CODE_REJECT_REASON'] =='HC').astype(int)
            self.df['NUM_REFUSED'] = (self.df['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
            
            agg_df = self.df.groupby(by='SK_ID_CURR').agg(
                    NUM_HC_REJECT_REASON_LOANS =('NUM_HC_REJECT_REASON','sum'),
                    NUM_REFUSED_LOANS =('NUM_REFUSED','sum'),
                ).reset_index()

            feature_df = self._create_ratio_feature(agg_df,'NUM_HC_REJECT_REASON_LOANS','NUM_REFUSED_LOANS','PA_RATIO_HC_REFUSED_LOANS')
    
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

            self.df['PA_NUM_SCO_SCOFR_REJECT_REASON'] = np.where(self.df['CODE_REJECT_REASON'].isin(['SCO','SCOFR']),
                                       1,
                                       0)
           
            feature_df =self.df.groupby(by='SK_ID_CURR')['PA_NUM_SCO_SCOFR_REJECT_REASON'].sum().to_frame().reset_index()

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

            self.df['PA_NUM_LIMIT_REJECT_REASON'] = (self.df['CODE_REJECT_REASON'] == 'LIMIT').astype(int)
           
            feature_df =self.df.groupby(by='SK_ID_CURR')['PA_NUM_LIMIT_REJECT_REASON'].sum().to_frame().reset_index()
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
            
            feature_df = self.df.groupby('SK_ID_CURR')['UNKNOWN_REJECT_REASON_CNT'].sum().to_frame('PA_UNKNOWN_REJECT_REASON_CNT').reset_index()
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

            self.df['FLAG_REPEATER'] = ( self.df['NAME_CLIENT_TYPE'] == 'Repeater').astype(int)

            feature_df = (
                 self.df.groupby('SK_ID_CURR')['FLAG_REPEATER']
                .mean()
                .to_frame('PA_RATIO_REPEATER')
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

            time_frames = [365,720,1080]
            features_df = pd.DataFrame()
            
            for frame in time_frames:

                self.df['FLAG_NEW_CLIENT'] = (
                    (self.df['DAYS_DECISION'] > -frame) &
                    (self.df['NAME_CLIENT_TYPE'] == 'New')
                ).astype(int)

                feature_df = (
                    self.df.groupby('SK_ID_CURR')['FLAG_NEW_CLIENT']
                    .max()
                    .to_frame(f'PA_FLAG_NEW_CLIENT_{frame}D')
                ).reset_index()

                if features_df.empty:
                    features_df = feature_df.copy()
                else:
                    features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')

            return features_df

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


            self.df['FLAG_HIGH_RISK_CHANNEL'] = self.df['CHANNEL_TYPE'].isin(
                ['AP+ (Cash loan)', 'Contact center']
            ).astype(int)

            agg_df = self.df.groupby('SK_ID_CURR').agg(
                NUM_HIGH_RISK_CHANNEL=('FLAG_HIGH_RISK_CHANNEL', 'sum'),
                TOTAL_APPLICATIONS=('FLAG_HIGH_RISK_CHANNEL', 'count')
            ).reset_index()
            
            feature_df = self._create_ratio_feature(agg_df,'NUM_HIGH_RISK_CHANNEL','TOTAL_APPLICATIONS','PA_RATIO_HIGH_RISK_CHANNEL')

            return feature_df

        else:
            logger.debug('Required columns  CHANNEL_TYPE not present in DataFrame')

    def _extract_ratio_insured_loans(self):

        ''' Extract features from the NFLAG_INSURED_ON_APPROVAL  in the previous application dataset.

            Features Transformed:
            - PA_RATIO_INSURED_LOANS: ratio of the insured loans of the client
            - PA_FLAG_EVER_INSURED_LOANS :  this client ever had the insured loan?
            - PA_INSURED_LAST_1Y : Recent Insurance Requests of  client within 1 year
            

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NFLAG_INSURED_ON_APPROVAL' in self.df.columns:
            
            feature_df = self.df.groupby('SK_ID_CURR')['NFLAG_INSURED_ON_APPROVAL'].agg(
                PA_RATIO_INSURED_LOANS = 'mean',
                PA_FLAG_EVER_INSURED_LOANS = 'max',
            ).reset_index()
            
            filt = (
                (self.df['DAYS_DECISION'] > -365) &
                (self.df['NFLAG_INSURED_ON_APPROVAL'] == 1)
            )

            recent_insured = (
                self.df.loc[filt]
                .groupby('SK_ID_CURR')['SK_ID_PREV']
                .nunique()
                .to_frame('PA_INSURED_LAST_1Y')
            )
            
            feature_df = feature_df.merge(recent_insured,on='SK_ID_CURR',how='outer')
            
        
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
            df = self.df.loc[~filt]
            
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

            agg_df = agg_df.rename(columns= {'FLAG_ALWAYS_UNACCOMPANIED':'PA_FLAG_ALWAYS_UNACCOMPANIED',
                                    'FLAG_NEVER_UNACCOMPANIED':'PA_FLAG_NEVER_UNACCOMPANIED',
                                    'RATIO_UNACCOMPANIED':'PA_RATIO_UNACCOMPANIED',
                                    'N_UNIQUE_SUITES':'PA_CNT_UNIQUE_SUITES'})
            
            features_df = agg_df[['SK_ID_CURR','PA_FLAG_ALWAYS_UNACCOMPANIED','PA_FLAG_NEVER_UNACCOMPANIED','PA_RATIO_UNACCOMPANIED','PA_CNT_UNIQUE_SUITES']]
            
            self.df['FLAG_FAMILY'] = (
                self.df['NAME_TYPE_SUITE']
                .isin(['Family','Spouse, partner'])
                ).astype(int)

            family_ratio = self.df.groupby('SK_ID_CURR')['FLAG_FAMILY'].mean().to_frame('PA_RATIO_FAMILY_ACCOMPANIED').reset_index()
            
            features_df = features_df.merge(family_ratio,on='SK_ID_CURR',how='outer')
            
                
                
            return features_df
    

        else:
            logger.debug('NAME_TYPE_SUITE column are not present in the DataFrame')


    def _extract_ratio_highrisk_yield_loans_d(self):

        ''' Extract ratio of the high_risk loans based on time frames 1,2,3,4 years

            Features Transformed:
            - RATIO_HIGH_RISK_YIELD_LOANS_XD: ratio high_risk loans / total loans time frames
            - PA_FLAG_HAD_HIGH_RISK_LOAN: flag if client ever had the high risk loan
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'NAME_YIELD_GROUP' in self.df.columns:
            time_frames = [360, 720, 1080] 
            features_df = pd.DataFrame()

            self.df['HIGH_RISK_YIELD_LOANS'] = (self.df['NAME_YIELD_GROUP']=='high').astype(int)

            for frame in time_frames:
                filt = self.df['DAYS_DECISION']> - frame
                filt_df = self.df.loc[filt]
                feature_df = filt_df.groupby('SK_ID_CURR')['HIGH_RISK_YIELD_LOANS'].mean().to_frame(f'PA_RATIO_HIGH_RISK_YIELD_LOANS_{frame}D')

                if features_df.empty:
                    features_df = feature_df.copy()
                else:
                    features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')            
            
            risk_loan = self.df.groupby(by='SK_ID_CURR')['HIGH_RISK_YIELD_LOANS'].max().to_frame('PA_FLAG_HAD_HIGH_RISK_LOAN').reset_index()
            features_df=  features_df.merge(risk_loan,how='outer',on='SK_ID_CURR')
            
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
                'XNA': np.nan
            }
            
            time_frames_days = [180, 360, 720, 1080]  
            features_df = pd.DataFrame()
            
            self.df['RISK_WEIGHT'] = self.df['NAME_YIELD_GROUP'].map(risk_map)

            for frame in time_frames_days:
                filt_df = self.df[self.df['DAYS_DECISION'] > -frame]
                # Group by client and calculate average
                avg_risk = (
                    filt_df.groupby('SK_ID_CURR')['RISK_WEIGHT']
                    .mean()
                    .to_frame(f'PA_AVG_RISK_WEIGHT_{frame}D')
                )
                flag_risk = (
                    filt_df.groupby('SK_ID_CURR')['RISK_WEIGHT']
                    .max()
                    .to_frame(f'PA_MAX_RISK_WEIGHT_{frame}D')
                )
                
                feature_df = avg_risk.merge(flag_risk,on='SK_ID_CURR',how='outer')
                
                if features_df.empty:
                    features_df = feature_df.copy()
                else:
                    features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')
                    
                
            features_df = features_df.reset_index()
            
            
            
            return features_df 
        
        else:
            logger.debug('NAME_YIELD_GROUP column is not present')

    def _extract_avg_max_loan_delay(self):
        '''
        Extract mean and max loan repayment delays per client and per loan category.

        Features Transformed:
            - PA_MEAN_LOAN_REPAYMENT_DIFF_<CATEGORY>
            - PA_MAX_LOAN_REPAYMENT_DIFF_<CATEGORY>
            - PA_MIN_LOAN_REPAYMENT_DIFF_<CATEGORY>
            Client-Level Features:
            - PA_AVG_LOAN_REPAYMENT_DIFF_CLIENT
            - PA_MAX_LOAN_REPAYMENT_DIFF_CLIENT
            - PA_MIN_LOAN_REPAYMENT_DIFF_CLIENT
            - PA_STD_LOAN_REPAYMENT_DIFF_CLIENT

        Returns:
            features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''

        if 'DAYS_TERMINATION' in self.df.columns and  'DAYS_LAST_DUE' in self.df.columns:

            self.df['DAYS_TERMINATION'] = self.df['DAYS_TERMINATION'].replace({365243.0:np.nan})
            self.df['DAYS_LAST_DUE'] = self.df['DAYS_LAST_DUE'].replace({365243.0:np.nan})
            
            categories = ['POS', 'Cash', 'Cards'] 
            features_df = pd.DataFrame()
            self.df['LOAN_REPAYMENT_DIFF'] = self.df['DAYS_TERMINATION'] - self.df['DAYS_LAST_DUE']
            
            for category in categories:
                filt = (self.df['NAME_PORTFOLIO'] == category)
                filt_df = self.df.loc[filt]

                agg_df = filt_df.groupby('SK_ID_CURR').agg(
                    MEAN_LOAN_REPAYMENT_DIFF = ('LOAN_REPAYMENT_DIFF','mean'),
                    MIN_LOAN_REPAYMENT_DIFF =  ('LOAN_REPAYMENT_DIFF','min'),
                    MAX_LOAN_REPAYMENT_DIFF =  ('LOAN_REPAYMENT_DIFF','max'),
                )
                
                agg_df = agg_df.rename(columns = {'MEAN_LOAN_REPAYMENT_DIFF':f'PA_MEAN_LOAN_REPAYMENT_DIFF_{category}',
                                         'MAX_LOAN_REPAYMENT_DIFF':f'PA_MAX_LOAN_REPAYMENT_DIFF_{category}',
                                         'MIN_LOAN_REPAYMENT_DIFF':f'PA_MIN_LOAN_REPAYMENT_DIFF_{category}'
                                         })
                
                if features_df.empty:
                    features_df = agg_df.copy()
                else:
                    features_df = features_df.merge(agg_df,on='SK_ID_CURR',how='outer')                
                    
            features_df = features_df.reset_index()
            
            pay_diff_client = self.df.groupby(by='SK_ID_CURR')['LOAN_REPAYMENT_DIFF'].agg(
                PA_AVG_LOAN_REPAYMENT_DIFF_CLIENT = 'mean',
                PA_MAX_LOAN_REPAYMENT_DIFF_CLIENT = 'max',
                PA_MIN_LOAN_REPAYMENT_DIFF_CLIENT = 'min',
                PA_STD_LOAN_REPAYMENT_DIFF_CLIENT = 'std',
                ).reset_index()
            
            features_df = features_df.merge(pay_diff_client,on='SK_ID_CURR',how='outer')
                
            return features_df
        
        else:
            logger.debug('DAYS_TERMINATION and  DAYS_LAST_DUE : column are not present in the DataFrame')

     
    def _extract_remaining_features(self):
        ''' feature are created after the baseline model'''
        pass
        
        
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
            

            for extractor in self.feature_extractors:
                 # log the method is running
                method_name = extractor.__name__
                logger.info(f"Current Method Running: {self.__class__.__name__}.{method_name}")                        
                       
                features_df = extractor()
                features_cols = features_df.columns.to_list()

                # log the method is running
               
                main_df = main_df.merge(features_df,on='SK_ID_CURR',how='left')
                main_df[features_cols] = main_df[features_cols].fillna(PLACEHOLDER)
                
            del features_df
            gc.collect()   
            logger.info("Aggregated features from Previous Application dataset dataframe successfully merged into the main  Application dataframe.")

            return main_df
         
        except Exception as e:
            raise MyException(e,sys,logger)



class CreditBalanceTransformation(BaseTransformer,RatioFeatureMixin):
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, data_indestion_artifact: DataIngestionArtifact):

        super().__init__(data_transformation_artifact, data_indestion_artifact)

        self.df = self.load_data(
            self.data_indestion_artifact.credit_card_balance,
            )

        self.time_frames = [1,3,6,9,12,18,24]
     
    @property 
    def monthly_credit_agg(self):
        '''
        Prepare reusable monthly-level credit card aggregates so dont have to calculate it multiple times.
        
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

    def _filter_months(self, df, frame):
        return df[(df['MONTHS_BALANCE'] >= -frame) & (df['MONTHS_BALANCE'] < 0)]


    def _extract_credit_utilization_features(self):

        ''' Extract maximum weighted credit utilization over multiple time frames.
            and trend of credit utilization 3M -> 12M

            WEIGHTED_UTILIZATION = sum(AMT_BALANCE) / sum(AMT_CREDIT_LIMIT_ACTUAL)

            Features Transformed:
            - CB_MAX_MONTHLY_UTIL_{XM}:- Maximum monthly credit utilization observed in the last X months
            - CB_UTIL_VOLATILITY_{XM}:Standard deviation of monthly utilization in the last X months
            - CB_WT_CREDIT_UTIL{XM} : Weighted Average of the credit utilization over the time window
            - CB_WT_CREDIT_UTIL_TREND_3M_12M: Difference between 3-month and 12-month weighted credit utilization,
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE'}

        if required_cols.issubset(self.df.columns):
            # Produces an average utilization per customer across their active lines.
            features_df = pd.DataFrame()

            monthly_df  = self.monthly_credit_agg
            

            for frame in self.time_frames:

                monthly_frame =self._filter_months(monthly_df,frame)
                
                #max or volatility features
                
                monthly_frame_temp = self._create_ratio_feature(monthly_frame,'AMT_BALANCE_TOTAL','AMT_CREDIT_LIMIT_ACTUAL_TOTAL','MONTHLY_UTIL')
                
                max_util = (
                    monthly_frame_temp
                    .groupby('SK_ID_CURR')['MONTHLY_UTIL']
                    .max()
                    .to_frame(f'CB_MAX_MONTHLY_UTIL_{frame}M')
                ).reset_index()
                

                std_util = (
                    monthly_frame_temp
                    .groupby('SK_ID_CURR')['MONTHLY_UTIL']
                    .std()
                    .to_frame(f'CB_UTIL_VOLATILITY_{frame}M')
                ).reset_index()
                                    
                # Weighted avg utilization features# 
                agg_df = (
                    monthly_frame
                    .groupby('SK_ID_CURR')
                    .agg(
                        BALANCE_SUM=('AMT_BALANCE_TOTAL', 'sum'),
                        LIMIT_SUM=('AMT_CREDIT_LIMIT_ACTUAL_TOTAL', 'sum')
                    )
                )
                agg_df = agg_df.reset_index()
                

                agg_df = self._create_ratio_feature(agg_df,'BALANCE_SUM','LIMIT_SUM','UTILIZATION_RATIO')
                
                agg_df = agg_df.rename(columns = {'UTILIZATION_RATIO':f'CB_WT_CREDIT_UTIL_{frame}M'})                
                # max_util    ,std_util agg_df
                frame_features = (
                    max_util
                    .merge(std_util, on='SK_ID_CURR', how='outer')
                    .merge(agg_df, on='SK_ID_CURR', how='outer')
                ) 
                if features_df.empty:
                    features_df = frame_features
                else:
                    features_df = features_df.merge(
                        frame_features,
                        on='SK_ID_CURR',
                        how='outer'
                    )
                
            features_df['CB_WT_CREDIT_UTIL_TREND_3M_12M'] = features_df['CB_WT_CREDIT_UTIL_3M'] - features_df['CB_WT_CREDIT_UTIL_12M']
            
            
            
            return features_df
        
        else:
            logger.debug("'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_BALANCE' column are not present in the DataFrame")

    def _extract_credit_usage_features(self):

        ''' Extract credit usage features based on AMT_DRAWINGS_CURRENT and AMT_CREDIT_LIMIT_ACTUAL
            from the Credit Balance dataset.
    
            Weighted monthly usage ratio is computed as:
            CREDIT_USAGE_RATIO = sum(AMT_DRAWINGS_CURRENT) / sum(AMT_CREDIT_LIMIT_ACTUAL)

            Features Transformed:
            - CB_MAX_CREDIT_USAGE_RATIO_{XM}:  Maximum monthly credit usage ratio over X months.
            - CB_WT_CREDIT_USAGE_{XM}: weighted credit usage over time frames
        
            - CB_TREND_CREDIT_DRAWING_3M_12M : Difference between 3-month and 12-month average usage ratios
                (indicates increasing or decreasing spending behavior).

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT', 'AMT_CREDIT_LIMIT_ACTUAL'}
        if required_cols.issubset(self.df.columns):
            features_df = pd.DataFrame()
            temp_df = pd.DataFrame()

            monthly_df  = self.monthly_credit_agg

            for frame in self.time_frames:

                monthly_frame = self._filter_months(monthly_df,frame)
                                
                
                new_monthly_frame = self._create_ratio_feature(
                    monthly_frame,
                    numerator='AMT_DRAWINGS_CURRENT_TOTAL',
                    denominator='AMT_CREDIT_LIMIT_ACTUAL_TOTAL',
                    feature_name='CREDIT_USAGE_RATIO_PER_MONTH'
                )
                
                #max ratio  for time frames
                max_ratio_df = (
                        new_monthly_frame.groupby('SK_ID_CURR')['CREDIT_USAGE_RATIO_PER_MONTH']
                        .max()
                        .to_frame(f'CB_MAX_CREDIT_USAGE_RATIO_{frame}M')
                        .reset_index()
                        )
                
                #weighted average for the time frames
                agg_df = (
                    monthly_frame.groupby('SK_ID_CURR')
                    .agg(
                        TOTAL_DRAWINGS=('AMT_DRAWINGS_CURRENT_TOTAL', 'sum'),
                        TOTAL_LIMIT=('AMT_CREDIT_LIMIT_ACTUAL_TOTAL', 'sum')
                    )
                )
                agg_df = agg_df.reset_index()
                agg_df = self._create_ratio_feature(agg_df, 'TOTAL_DRAWINGS', 'TOTAL_LIMIT', f'CB_WT_CREDIT_USAGE_{frame}M')

                if frame in [3,12]:
                    feature_df = new_monthly_frame.groupby('SK_ID_CURR')['CREDIT_USAGE_RATIO_PER_MONTH'].mean().to_frame(f'AVG_CREDIT_DRAWING_RATIO_{frame}M').reset_index()
                    if features_df.empty:
                        features_df = feature_df
                    else:
                        features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')

                #max_ratio_df, agg_df
                
                frame_features = (
                    max_ratio_df
                    .merge(agg_df, on='SK_ID_CURR', how='outer')
                ) 
                if temp_df.empty:
                    temp_df = frame_features
                else:
                    temp_df = temp_df.merge(
                        frame_features,
                        on='SK_ID_CURR',
                        how='outer'
                    )
       
                
            features_df['CB_TREND_CREDIT_DRAWING_3M_12M'] = features_df['AVG_CREDIT_DRAWING_RATIO_3M'] - features_df['AVG_CREDIT_DRAWING_RATIO_12M']
            features_df = features_df.merge(temp_df,on='SK_ID_CURR',how='left')
 

            return features_df
        else:
            logger.debug("'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL',AMT_DRAWINGS_CURRENT  column are not present in the DataFrame")

    def _extract_atm_cash_usage_features(self):

        '''  Extract ATM cash utilization features from the Credit Card Balance dataset.

            Features Transformed:
            - CB_MAX_RATIO_ATM_CASH_UTILIZATION__{XM}:  Maximum ATM cash monthly usage ratio over X months.
            - CB_TREND_ATM_CASH_UTILIZATION_3M_12M: Difference between 3-month and 12-month average ATM cash utilization.
                     Positive value indicates increasing reliance on ATM cash.
            - CB_WT_ATM_CASH_UTIL_{XM}: Weighted Atm Cash utilization over x months
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_CREDIT_LIMIT_ACTUAL'}

        if required_cols.issubset(self.df.columns):
            features_df = pd.DataFrame()
            temp_df = pd.DataFrame()

            monthly_df  = self.monthly_credit_agg

            for frame in self.time_frames:

                monthly_frame =self._filter_months(monthly_df,frame)
                

                frame_ratio_df = self._create_ratio_feature(
                            monthly_frame,
                            numerator='AMT_DRAWINGS_ATM_CURRENT_TOTAL',
                            denominator='AMT_CREDIT_LIMIT_ACTUAL_TOTAL',
                            feature_name='ATM_CASH_UTILIZATION_RATIO'
                        )
                #max ratios dataframe
                
                max_ratio_df = (
                        frame_ratio_df.groupby('SK_ID_CURR')['ATM_CASH_UTILIZATION_RATIO']
                        .max()
                        .to_frame(f'CB_MAX_RATIO_ATM_CASH_UTILIZATION_{frame}M')
                        .reset_index()
                    )
                
                # Weighted utilization for this frame
                agg_df = (
                    monthly_frame.groupby('SK_ID_CURR')
                    .agg(
                        TOTAL_ATM_DRAWINGS=('AMT_DRAWINGS_ATM_CURRENT_TOTAL', 'sum'),
                        TOTAL_LIMIT=('AMT_CREDIT_LIMIT_ACTUAL_TOTAL', 'sum')
                    )
                )
                agg_df = agg_df.reset_index()
                agg_df = self._create_ratio_feature(agg_df, 'TOTAL_ATM_DRAWINGS', 'TOTAL_LIMIT', f'CB_WT_ATM_CASH_UTIL_{frame}M')
                                
                if frame in [3,12]:
                    avg_df  = frame_ratio_df.groupby('SK_ID_CURR')['ATM_CASH_UTILIZATION_RATIO'].mean().to_frame(f'AVG_ATM_CASH_UTILIZATION_RATIO_{frame}M').reset_index()
                    if features_df.empty:
                        features_df = avg_df
                    else:
                        features_df = features_df.merge(avg_df,on='SK_ID_CURR' ,how='outer')

                frame_features = max_ratio_df.merge(agg_df, on='SK_ID_CURR', how='outer')
                                 
                if temp_df.empty:
                    temp_df = frame_features
                else:
                    temp_df = temp_df.merge(
                        frame_features,
                        on='SK_ID_CURR',
                        how='outer'
                    )
                    
            features_df['CB_TREND_ATM_CASH_UTILIZATION_3M_12M'] = features_df['AVG_ATM_CASH_UTILIZATION_RATIO_3M'] - features_df['AVG_ATM_CASH_UTILIZATION_RATIO_12M']
            
             # Merge all max + weighted features
            features_df = features_df.merge(temp_df, on='SK_ID_CURR', how='left')


            return features_df
        
        else:
            logger.debug("'MONTHS_BALANCE', 'AMT_DRAWINGS_ATM_CURRENT' AMT_CREDIT_LIMIT_ACTUAL ,  column are not present in the DataFrame")


    def _extract_avg_atm_withdrawal_frequency(self):
    
        '''  Extract the avg frequency of the atm withdrawl of  client in time frames features from the Credit Card Balance dataset.


            Features Transformed:
            - CB_AVG_ATM_WITHDRAWAL_FREQ_{XM}: Average number of ATM withdrawals per active time window

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'CNT_DRAWINGS_ATM_CURRENT'}

        if required_cols.issubset(self.df.columns):
            features_df = pd.DataFrame()
            for frame in self.time_frames:

                monthly_frame =self._filter_months(self.df,frame)


                feature_df = (
                    monthly_frame
                    .groupby('SK_ID_CURR')['CNT_DRAWINGS_ATM_CURRENT']
                    .mean()
                    .to_frame(f'CB_AVG_ATM_WITHDRAWAL_FREQ_{frame}M')
                ).reset_index()
                if features_df.empty:
                    features_df = feature_df
                else:
                    features_df  =features_df.merge(feature_df,on='SK_ID_CURR',how='outer')


            return features_df
        else:
            logger.debug(f'{required_cols} are not present in dataframe')

#-----------------------------------------------
    def _extract_pos_utilization_features(self):

        ''' Extract maximum POS spending utilization features over multiple time windows
                from the Credit Card Balance dataset.

            Features Transformed:
            - CB_RATIO_MAX_POS_SPEND_{XM}: Maximum ratio of POS spending to total credit limit over the last X months.
            - CB_MAX_RATIO_POS_TO_TOTAL_DRAW_{XM}:  Maximum ratio of POS spending to total monthly spend over the last X months.

            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_DRAWINGS_POS_CURRENT','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_CURRENT'}
        if required_cols.issubset(self.df.columns):
            
            features_df = pd.DataFrame()
            
            monthly_df  = self.monthly_credit_agg

        
            for frame in self.time_frames:
                

                monthly_frame =self._filter_months(monthly_df,frame)

                
                df_1 = self._create_ratio_feature(monthly_frame,
                                                    'AMT_DRAWINGS_POS_CURRENT_TOTAL',
                                                    'AMT_CREDIT_LIMIT_ACTUAL_TOTAL',
                                                    'CB_RATIO_POS_SPEND_PER_MONTH')
                df_2 = self._create_ratio_feature(monthly_frame,
                                                    'AMT_DRAWINGS_POS_CURRENT_TOTAL',
                                                    'AMT_DRAWINGS_CURRENT_TOTAL',
                                                    'CB_RATIO_POS_TO_TOTAL_DRAW')
                
                feature_df_1 = df_1.groupby(by='SK_ID_CURR')['CB_RATIO_POS_SPEND_PER_MONTH'].max().to_frame(f'CB_RATIO_MAX_POS_SPEND_{frame}M')
                feature_df_2 = df_2.groupby(by='SK_ID_CURR')['CB_RATIO_POS_TO_TOTAL_DRAW'].max().to_frame(f'CB_MAX_RATIO_POS_TO_TOTAL_DRAW_{frame}M')


                feature_df_list = [feature_df_1,feature_df_2]
                frame_features  = pd.concat(feature_df_list,axis=1)

                if features_df.empty:
                    features_df = frame_features
                else:
                    features_df = pd.concat([features_df, frame_features], axis=1)
            features_df = features_df.reset_index()


            return features_df
        
        else:
            logger.debug(f"{required_cols} column are not present in the DataFrame")
    #-----------------------------------------------------------------------
    def _extract_payment_behavior_features(self):
        ''' Extracts payment behavior features from the Credit Card Balance dataset over multiple time windows.


            Features Transformed:
            - CB_MAX_RATIO_AMT_PAYMENT_MIN_INST_:  Max ratio of payments made to minimum required payments over the last X months
            - CB_RATIO_UNDERPAYMENT_:  Ratio of months where actual payment < minimum required payment  over the last X months
            - CB_STD_PAYMENT_VOLATILITY_: Standard deviation of payment-to-mini  mum-installment ratio over the last X months
            - CB_MAX_RATIO_PAYMENT_BALANCE_: Max payment-to-balance ratio over last X months
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = {'MONTHS_BALANCE', 'AMT_INST_MIN_REGULARITY','AMT_PAYMENT_CURRENT','AMT_BALANCE'}
        if required_cols.issubset(self.df.columns):

            features_df = pd.DataFrame()
            monthly_df  = self.monthly_credit_agg

        
            for frame in self.time_frames:

                monthly_frame =self._filter_months(monthly_df,frame)

                # payment-to-minimum-installment ratio
                df_1 = self._create_ratio_feature(monthly_frame,
                                                  'AMT_PAYMENT_CURRENT_TOTAL',
                                                  'AMT_INST_MIN_REGULARITY_TOTAL',
                                                  'CB_RATIO_AMT_PAYMENT_MIN_INST')
                
                df_2 = self._create_ratio_feature(monthly_frame,
                                                  'AMT_PAYMENT_CURRENT_TOTAL',
                                                  'AMT_BALANCE_TOTAL',
                                                  'CB_RATIO_PAYMENT_TO_BALANCE')
                
              
                
                #underpayment flag
                monthly_frame['UNDERPAYMENT_FLAG'] = (monthly_frame['AMT_PAYMENT_CURRENT_TOTAL'] < monthly_frame['AMT_INST_MIN_REGULARITY_TOTAL']).astype(int)
                
                # max payment / min installmetn ratio
                feature_df_1=  df_1.groupby(by='SK_ID_CURR')['CB_RATIO_AMT_PAYMENT_MIN_INST'].max().to_frame(f'CB_MAX_RATIO_AMT_PAYMENT_MIN_INST_{frame}M').reset_index()

                #underpayment ratio
                feature_df_2 = monthly_frame.groupby(by='SK_ID_CURR')['UNDERPAYMENT_FLAG'].mean().to_frame(f'CB_RATIO_UNDERPAYMENT_{frame}M').reset_index()
                
                # payment volatility (standard deviation)
                feature_df_3=  df_1.groupby(by='SK_ID_CURR')['CB_RATIO_AMT_PAYMENT_MIN_INST'].std().to_frame(f'CB_STD_PAYMENT_VOLATILITY_{frame}M').reset_index()
                
                # PAYMENT_TO_BALANCE_RATIO
                feature_df_4 = df_2.groupby('SK_ID_CURR')['CB_RATIO_PAYMENT_TO_BALANCE'].max().to_frame(f'CB_MAX_RATIO_PAYMENT_BALANCE_{frame}M').reset_index()

                if features_df.empty:
                    features_df = (
                        feature_df_1.merge(feature_df_2,on='SK_ID_CURR',how='outer')
                        .merge(feature_df_3,on='SK_ID_CURR',how='outer')
                        .merge(feature_df_4,on='SK_ID_CURR',how='outer')
                            )
                else:
                    features_df = (
                        features_df.merge(feature_df_1,on='SK_ID_CURR',how='outer')
                        .merge(feature_df_2,on='SK_ID_CURR',how='outer')
                        .merge(feature_df_3,on='SK_ID_CURR',how='outer')
                        .merge(feature_df_4,on='SK_ID_CURR',how='outer')    )
                    
    

            return features_df
        
        else:
            logger.debug(" MONTHS_BALANCE', 'AMT_INST_MIN_REGULARITY','AMT_PAYMENT_CURRENT' column are not present in the DataFrame")
        
#-----------------------------------------------------------------------

    def _extract_credit_utilization_ratios(self):

        ''' Extract features from the, 'AMT_RECEIVABLE_PRINCIPAL','AMT_CREDIT_LIMIT_ACTUAL' in the Credit Balance dataset.

            Features Transformed:
            - CB_MAX_RATIO_PRINCIPAL_UTILIZATION_{X}M: MAX Average ratio of receivable principal to credit limit in las x months.
           
            Returns:
                features_df(pd.DataFrame): DataFrame with SK_ID_CURR Index and Transformed features
        '''
        required_cols = { 'AMT_RECEIVABLE_PRINCIPAL','AMT_CREDIT_LIMIT_ACTUAL','MONTHS_BALANCE'}
        if required_cols.issubset(self.df.columns):

            features_df = pd.DataFrame()
            monthly_df  = self.monthly_credit_agg


            for frame in self.time_frames:
                monthly_frame =self._filter_months(monthly_df,frame)

    
                #represents how much principal amount of their credit limit is currently used.
                new_df = self._create_ratio_feature(monthly_frame,
                                           'AMT_RECEIVABLE_PRINCIPAL_TOTAL',
                                           'AMT_CREDIT_LIMIT_ACTUAL_TOTAL',
                                           'CB_RATIO_PRINCIPAL_UTILIZATION')
                
                feature_df = new_df.groupby(by='SK_ID_CURR')['CB_RATIO_PRINCIPAL_UTILIZATION'].max().to_frame(f'CB_MAX_RATIO_PRINCIPAL_UTILIZATION_{frame}M').reset_index()
                if features_df.empty:
                    features_df = feature_df
                else:
                    features_df = features_df.merge(feature_df,on='SK_ID_CURR',how='outer')
            

            return features_df
        
        else:
            logger.debug(f"{required_cols} column are not present in the DataFrame")

#-------------------------------------------------------------------------        
  
    def _extract_worst_dpd_features_credit(self):
        '''  Create worst DPD (Days Past Due) features for multiple time frames 
            from the credit balance dataset
    
            Features Extracted:
            - CB_WORST_DPD_{XM}: based on time frame:[3, 6, 9, 12, 24, 36, 72, 96] M

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as index and WORST_DPD_CREDIT_XM features
                Missing values filled with the placeholder -66666

        '''
        if 'SK_DPD' in self.df.columns and 'MONTHS_BALANCE' in self.df.columns:
            time_frames = [3, 6, 9, 12, 24, 36, 72, 96]
            # empty dataframe to apend the feature into
            features_df = pd.DataFrame()

            for frame in time_frames:
                monthly_frame =self._filter_months(self.df,frame)

                feature_dpd = monthly_frame.groupby(by='SK_ID_CURR')['SK_DPD'].max().to_frame(f'CB_WORST_DPD_{frame}M').reset_index()
                if features_df.empty:
                    features_df = feature_dpd
                else:
                    features_df = features_df.merge(feature_dpd,on='SK_ID_CURR',how='outer')

            features_df = features_df.fillna(DPD_LOAN_DATA_MISSING) # -66666


            return features_df
        else:
            logger.debug('SK_DPD : column is not present in the DataFrame')

    def _extract_worst_dpd_def_features_credit(self):
        '''Create severe DPD (Days Past Due with tolerance) features for multiple time frames 
            from the Credit balance dataset.

            Features Extracted:
            - CB_WORST_DPD_DEF_{XM} for time frames [3, 6, 9, 12, 24, 36, 72, 96]

            Returns:
            - feature_df : 
                DataFrame with SK_ID_CURR as index and WORST_DPD_DEF_POS_CASH_XM features
                Missing values filled with the placeholder -66666
            '''
        if 'SK_DPD_DEF' in self.df.columns and 'MONTHS_BALANCE' in self.df.columns:
            time_frames = [3, 6, 9, 12, 24, 36, 72, 96]
            # empty dataframe to apend the feature into
            features_df = pd.DataFrame()

            for frame in time_frames:
                monthly_frame =self._filter_months(self.df,frame)

                
                feature_dpd = monthly_frame.groupby(by='SK_ID_CURR')['SK_DPD_DEF'].max().to_frame(f'CB_WORST_DPD_DEF_{frame}M').reset_index()
                
                if features_df.empty:
                    features_df = feature_dpd
                else:
                    features_df = features_df.merge(feature_dpd,on='SK_ID_CURR',how='outer')

            features_df = features_df.fillna(DPD_LOAN_DATA_MISSING) # -66666
            
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
                 # log the method is running
                method_name = extractor.__name__
                logger.info(f"Current Method Running: {self.__class__.__name__}.{method_name}")        

                
                features_df = extractor()
                
       
                features_cols = features_df.columns.to_list()

                main_df = main_df.merge(features_df,on='SK_ID_CURR',how='left')
                main_df[features_cols] = main_df[features_cols].fillna(PLACEHOLDER)
                
            del features_df
            gc.collect()   
            logger.info("Aggregated features from Credit card balance dataframe successfully merged into the main  Application dataframe.")

            return main_df
         
        except Exception as e:
            logger.exception("Error while adding credit card balance features")
            raise MyException(e,sys,logger)

      
class DataTransformation:
     
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,data_indestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact):
        self.data_transformation_artifact = data_transformation_artifact
        self.data_indestion_artifact = data_indestion_artifact 
        self.data_validation_artifact = data_validation_artifact
            
        self.main_df_path = self.data_indestion_artifact.application_data
        
        self.output_path = self.data_transformation_artifact.artifact_interim_main_transformed_df
        
        self.feature_transformers =  [
            BureauBalanceTransformation,
            BureauTransformer,
            InstallmentsPaymentsTransformation,
            PosCashBalanceTransformation,
            PreviousApplicationsTransformation,
            CreditBalanceTransformation
            ]
        
    def _is_data_validated(self):
        '''check and load status of the data validation from yaml file
            
            return:
                True | False : status of data validation
        '''

        file = read_yaml_file(self.data_validation_artifact.validation_report_path,logger)
        
        return  file.get('is_data_validated', False)
    
    
    def _load_and_prepare_main_df(self) -> pd.DataFrame:
        """
        Load and preprocess the main application dataframe.
        """
        logger.info("Loading and preprocessing main application dataframe")

        transformer = ApplicationDfTransformer(
            self.main_df_path,
            DataIngestionArtifact()
        )

        return transformer.run_preprocessing_steps()
    def _save_transformed_data(self, df: pd.DataFrame) -> None:
        """
        Save transformed dataframe artifact/interim.
        """
        
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
                self.data_transformation_artifact,
                self.data_indestion_artifact
            )
            main_df = transformer.add_features_main(main_df)
            float_cols = main_df.select_dtypes(include=['float64']).columns
            main_df[float_cols] = main_df[float_cols].astype('float32')
            
            del transformer
            gc.collect()


        return main_df 
       
    def run(self):
        if self._is_data_validated():
            logger.info('DATA IS VALIDATED')
            try:
                main_df = self._load_and_prepare_main_df()
                # did this on purpose to handle this manually for improve this later
    
                final_df = self._apply_feature_transformations(main_df)
                self._save_transformed_data(final_df)
            
                logger.info('DATA TRANSFORMATION COMPLETED  SUCCESSFULLY')

            except Exception as e:
                raise MyException(e,sys,logger)
        else:
            logger.error("Data validation failed. Transformation Didnt happened.")

 
if __name__ =='__main__':
    
    data_transformation_artifact = DataTransformationArtifact()
    data_indestion_artifact = DataIngestionArtifact()
    data_validation_artifact = DataValidationArtifact()
    data_transformation = DataTransformation(
        data_transformation_artifact=data_transformation_artifact,
        data_indestion_artifact=data_indestion_artifact,
        data_validation_artifact=data_validation_artifact
    )

    data_transformation.run()


   
