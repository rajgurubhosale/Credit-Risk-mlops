import os

ARTIFACT_DIR = r'artifact'
SPLIT_DIR = os.path.join(ARTIFACT_DIR,'splits')

BINNING_DIR = os.path.join(ARTIFACT_DIR,'binning')
AUTOMATIC_BINNING_DIR = os.path.join(BINNING_DIR,'automatic')

# Missing value handling
DEFAULT_NUM_MISSING_VALUE = -99999

# Special codes (global)
SPECIAL_CODES = {
    "SC_-99999": [-99999],
    "SC_-88888": [-88888],
    "SC_-77777": [-77777],
}

# Random seed
DEFAULT_RANDOM_STATE = 42
TEST_SIZE = 0.30
