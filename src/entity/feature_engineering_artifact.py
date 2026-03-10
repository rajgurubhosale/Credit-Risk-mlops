from dataclasses import dataclass
from src.constants.feature_engineering_constant import *


@dataclass
class BinningConfig:
    # Numerical binning
    max_n_bins: int = 20
    min_bin_pct: float = 0.05
    prebinning_method: str = "quantile"
    solver: str = "cp"

    # Monotonicity (future)
    monotonic_trend: str | None = None  # "ascending", "descending", None

    # Categorical binning
    rare_threshold: float = 0.01
    eps: float = 1e-6

class FeatureEngConfig:
    default_num_missing_value:int = DEFAULT_NUM_MISSING_VALUE
    special_codes:dict = SPECIAL_CODES
    default_random_state:int = DEFAULT_RANDOM_STATE 
    test_size:float = TEST_SIZE
    splits_dir = SPLIT_DIR
    bin_dir = BINNING_DIR
    prebin_dir = PREBIN_DIR
    
