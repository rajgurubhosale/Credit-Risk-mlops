import numpy as np



DATA_VALIDATION_YAML = 'D:/home loan credit risk/validation_report/validation_report.yaml'


''' MAPPINGS'''

PLACEHOLDERS = {
    'local_placeholder': 
    {
        'DAYS_EMPLOYED':{365243:np.nan}
    },
    
    'global_placeholders':{"XNA": np.nan, "XAP": np.nan, "Unknown": np.nan}
}
SIMPLIFY_VALUES = {
'NAME_EDUCATION_TYPE':{'Secondary / secondary special':'Secondary education'},
'NAME_FAMILY_STATUS':{'Single / not married': 'Single'},
'NAME_HOUSING_TYPE':{ 'House / apartment': 'Owned'}
}

DAYS_TO_YEARS_MAPPING = {
        "DAYS_BIRTH": "YEARS_AGE",
        "DAYS_EMPLOYED": "YEARS_EMPLOYED",
        "DAYS_REGISTRATION": "YEARS_REGISTRATION",
        "DAYS_ID_PUBLISH": "YEARS_ID_PUBLISH"
}



'''data types reduced '''
APPLICATION_DATA_DTYPES_REDUCE = {
    "FLAG_DOCUMENT_1":'int8',
    "FLAG_DOCUMENT_2":'int8',
    "FLAG_DOCUMENT_3":'int8',
    "FLAG_DOCUMENT_4":'int8',	
    "FLAG_DOCUMENT_5":'int8',	
    "FLAG_DOCUMENT_6":'int8',	
    "FLAG_DOCUMENT_7":'int8',	
    "FLAG_DOCUMENT_8":'int8',
    "FLAG_DOCUMENT_9":'int8',	
    "FLAG_DOCUMENT_10":'int8',	
    "FLAG_DOCUMENT_11":'int8',	
    "FLAG_DOCUMENT_12":'int8',	
    "FLAG_DOCUMENT_13":'int8',	
    "FLAG_DOCUMENT_14":'int8',	
    "FLAG_DOCUMENT_15":'int8',	
    "FLAG_DOCUMENT_16":'int8',	
    "FLAG_DOCUMENT_17":'int8',	
    "FLAG_DOCUMENT_18":'int8',	
    "FLAG_DOCUMENT_19":'int8',	
    "FLAG_DOCUMENT_20":'int8',	
    "FLAG_DOCUMENT_21":'int8',
    "FLAG_MOBIL":'int8',
    "FLAG_EMP_PHONE":'int8',
    "FLAG_WORK_PHONE":'int8',
    "FLAG_CONT_MOBILE":'int8',
    "FLAG_PHONE":'int8',
    "FLAG_EMAIL":'int8',
    "REGION_RATING_CLIENT":'int8',
    "REGION_RATING_CLIENT_W_CITY":'int8',



    "DAYS_BIRTH":'int32',
    "DAYS_EMPLOYED":'int32',
    "DAYS_ID_PUBLIS":'int32',
    "HOUR_APPR_PROCESS_START":'int32',
    "REG_REGION_NOT_LIVE_REGION":'int32',
    "REG_REGION_NOT_WORK_REGION":'int32',
    "LIVE_REGION_NOT_WORK_REGION":'int32',
    "REG_CITY_NOT_LIVE_CITY":'int32',
    "REG_CITY_NOT_WORK_CITY":'int32',
    "LIVE_CITY_NOT_WORK_CITY":'int32',


    "OWN_CAR_AGE":'float32',
    "DAYS_REGISTRATION":'float32',
    "CNT_FAM_MEMBERS":'float32',
    "AMT_REQ_CREDIT_BUREAU_DAY":'float32',
    "AMT_REQ_CREDIT_BUREAU_WEEK":'float32',
    "AMT_REQ_CREDIT_BUREAU_MON":'float32',
    "AMT_REQ_CREDIT_BUREAU_QRT":'float32',
    "AMT_REQ_CREDIT_BUREAU_YEAR":'float32',
    "AMT_REQ_CREDIT_BUREAU_HOUR":'float32',
    "REGION_POPULATION_RELATIVE":'float32',
    "OBS_30_CNT_SOCIAL_CIRCLE":'float32',
    "DEF_30_CNT_SOCIAL_CIRCLE":'float32',
    "OBS_60_CNT_SOCIAL_CIRCLE":'float32',
    "DEF_60_CNT_SOCIAL_CIRCLE":'float32',
    "DAYS_LAST_PHONE_CHANGE":'float32',
    "APARTMENTS_AVG":'float32',
    "BASEMENTAREA_AVG":'float32',
    "YEARS_BEGINEXPLUATATION_AVG":'float32',
    "YEARS_BUILD_AVG":'float32',
    "COMMONAREA_AVG":'float32',
    "ELEVATORS_AVG":'float32',
    "ENTRANCES_AVG":'float32',
    "FLOORSMAX_AVG":'float32',
    "FLOORSMIN_AVG":'float32',
    "LANDAREA_AVG":'float32',
    "LIVINGAPARTMENTS_AVG":'float32',
    "LIVINGAREA_AVG":'float32',
    "NONLIVINGAPARTMENTS_AVG":'float32',
    "NONLIVINGAREA_AVG":'float32',
    "APARTMENTS_MODE":'float32',
    "BASEMENTAREA_MODE":'float32',
    "YEARS_BEGINEXPLUATATION_MODE":'float32',
    "YEARS_BUILD_MODE":'float32',
    "COMMONAREA_MODE":'float32',
    "ELEVATORS_MODE":'float32',
    "ENTRANCES_MODE":'float32',
    "FLOORSMAX_MODE":'float32',
    "FLOORSMIN_MODE":'float32',
    "LANDAREA_MODE":'float32',
    "LIVINGAPARTMENTS_MODE":'float32',
    "LIVINGAREA_MODE":'float32',
    "NONLIVINGAPARTMENTS_MODE":'float32',
    "NONLIVINGAREA_MODE":'float32',
    "APARTMENTS_MEDI":'float32',
    "BASEMENTAREA_MEDI":'float32',
    "YEARS_BEGINEXPLUATATION_MEDI":'float32',
    "YEARS_BUILD_MEDI":'float32',
    "COMMONAREA_MEDI":'float32',
    "ELEVATORS_MEDI":'float32',
    "ENTRANCES_MEDI":'float32',
    "FLOORSMAX_MEDI":'float32',
    "FLOORSMIN_MEDI":'float32',
    "LANDAREA_MEDI":'float32',
    "LIVINGAPARTMENTS_MEDI":'float32',
    "LIVINGAREA_MEDI":'float32',
    "NONLIVINGAPARTMENTS_MEDI":'float32',
    "NONLIVINGAREA_MEDI":'float32',
    "TOTALAREA_MODE":'float32'}


BUREAU_DTYPES_REDUCE ={
    'DAYS_CREDIT':'int16',
    'CREDIT_DAY_OVERDUE':'int16',
    'CNT_CREDIT_PROLONG':'int16',
    
    'DAYS_CREDIT_UPDATE':'int32',
    
    'DAYS_CREDIT_ENDDATE':'float32',
    'DAYS_ENDDATE_FACT':'float32',
}

BUREAU_BALANCE_DTYPES_REDUCE = {
    'SK_ID_BUREAU':'int32',
    'MONTHS_BALANCE':'int16',
}

INSTALLMENT_PAYMENTS_DTYPES_REDUCE = {
    'NUM_INSTALMENT_VERSION': 'float32',
    'NUM_INSTALMENT_NUMBER':'int16',
    'DAYS_INSTALMENT':'float32',
    'DAYS_ENTRY_PAYMENT':'float32',
    'DPD':'int32',
    'MEAN_DPD_LATE_ONLY':'float32'

}

PREVIOUS_APPLICATION_DTYPES_REDUCE ={
    'HOUR_APPR_PROCESS_START':'int16',
    'NFLAG_LAST_APPL_IN_DAY	':'int8',


    'RATE_DOWN_PAYMENT':'float32',
    'RATE_INTEREST_PRIMARY':'float32',
    'RATE_INTEREST_PRIVILEGED':'float32',


    'DAYS_DECISION':'int32',
    'SELLERPLACE_AREA':'int32',


    'CNT_PAYMENT':'float32',
    'DAYS_FIRST_DRAWING':'float32',
    'DAYS_FIRST_DUE':'float32',
    'DAYS_LAST_DUE_1ST_VERSION':'float32',
    'DAYS_LAST_DUE':'float32',
    'DAYS_TERMINATION':'float32',
    'NFLAG_INSURED_ON_APPROVAL':'float32'
    
}


POS_CASH_REDUCE_DTYPES = {
    'MONTH_BALANCE':'int16',
    'CNT_INSTALMENT':'float32',
    'CNT_INSTALMENT_FUTURE':'float32',
    'SK_DPD':'int16',
    'SK_DPD_DEF':'int16'
}
## transformed main_df features dtypes:
TRANSFORMED_DF_DTYPES_REDUCE = {}
CREDIT_CARD_BALANCE_REDUCE_DTYPES = {
    'MONTHS_BALANCE':'int16',
    'CNT_DRAWINGS_ATM_CURRENT':'float32',
    'CNT_DRAWINGS_CURRENT':'int16',
    'CNT_DRAWINGS_OTHER_CURRENT':'float32',
    'CNT_DRAWINGS_POS_CURRENT':'float32',
    'CNT_INSTALMENT_MATURE_CUM':'float32',
    'SK_DPD':'int16',
    'SK_DPD_DEF':'int16'
}