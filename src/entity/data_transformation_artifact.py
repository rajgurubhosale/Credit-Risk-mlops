from src.constants.data_transformation_constant import *
from src.utils.main_utils import *
from dataclasses import dataclass, field
from src.utils.main_utils import *
from src.utils.main_utils import read_yaml_file
import numpy as np


@dataclass
class ApplicationDfConfig:
    simplify_values_mapping: dict = field(default_factory=lambda: SIMPLIFY_VALUES)
    placeholders_mapping: dict = field(default_factory=lambda: PLACEHOLDERS)
    days_to_years_mapping: dict = field(default_factory=lambda: DAYS_TO_YEARS_MAPPING)
    application_data_dtypes_reduce: dict = field(default_factory=lambda: APPLICATION_DATA_DTYPES_REDUCE)



@dataclass    
class DataTransformationConfig:
    data_validation_yaml: dict = field(default_factory=lambda: DATA_VALIDATION_YAML)
    bureau_dtypes_reduce: dict = field(default_factory=lambda: BUREAU_DTYPES_REDUCE)
    bureau_balance_dtypes_reduce: dict = field(default_factory=lambda: BUREAU_BALANCE_DTYPES_REDUCE)
    installment_payment_dtypes_reduce: dict = field(default_factory=lambda: INSTALLMENT_PAYMENTS_DTYPES_REDUCE)
    previous_application_dtypes_reduce: dict = field(default_factory=lambda: PREVIOUS_APPLICATION_DTYPES_REDUCE)
    pos_cash_reduce_dtypes: dict = field(default_factory=lambda: POS_CASH_REDUCE_DTYPES)
    credit_card_balance_reduce_dtypes: dict = field(default_factory=lambda: CREDIT_CARD_BALANCE_REDUCE_DTYPES)
    aggregated_artifact_interim_dir: str = ARTIFACT_INTERIM_DIR
