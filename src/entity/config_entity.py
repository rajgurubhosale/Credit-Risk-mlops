# WHAT NEEDS TO RUN THE STAGES PUT THAT ONLY HERE
from pathlib import Path
from src.constants.artifacts_paths import *
from dataclasses import dataclass, field
from src.utils.main_utils import read_yaml_file
from src.constants.bin_refinement_manual_constant import BIN_MERGING_PLANS

@dataclass
class DataIngestionConfig:
    source_raw_data_url:Path = SOURCE_RAW_DATA_URL
    source_data_names_list: list = field(default_factory=lambda: DATABASE_NAMES_LIST)

@dataclass
class FeatureQualityFilterConfig:
    params_path: Path = PARAMS_DIR_PATH
    
   

@dataclass
class FeatureEngConfig:
    params_path: Path = PARAMS_DIR_PATH
    special_codes:dict = field(default_factory=lambda: SPECIAL_CODES)

   

@dataclass
class FeatureBinMergingConfig:
     bin_merging_plans: dict =field(default_factory=lambda: BIN_MERGING_PLANS)

