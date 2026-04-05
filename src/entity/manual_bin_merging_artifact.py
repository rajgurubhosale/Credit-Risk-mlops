from src.constants.manual_bin_merging import *
from src.utils.main_utils import *
from dataclasses import dataclass, field
from src.utils.main_utils import *

@dataclass
class PostBinMergingConfig:
    bin_merging_plans: dict = field(default_factory=lambda: BIN_MERGING_PLANS)
    artifact_final_dir: dict = field(default_factory=lambda: ARTIFACT_FINAL_DIR)