from dataclasses import dataclass, field
from src.constants.data_quality_check_constant import *


@dataclass
class DataQualityConfig:
    threshold: float = MAX_BAD_RATIO_THRESHOLD
    special_values: list = field(default_factory=lambda: SPECIAL_VALUES)
    data_quality_dir:str = DATA_QUALITY_DIR
