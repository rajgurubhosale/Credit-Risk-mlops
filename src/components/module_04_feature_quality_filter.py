import os
import pandas as pd
from pathlib import Path
from src.entity.config_entity import FeatureQualityFilterConfig
from src.entity.artifact_entity import FeatureQualityFilterArtifact
from src.logger import config_logger
from src.utils.main_utils import read_yaml_file
from src.constants.artifacts_paths import * 
# constants

logger = config_logger('module_04_feature_quality_filter.py')



def  generate_feature_quality_df(df,special_values,max_threshold):
    """Create feature quality report"""

    # downcast
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    special_df = pd.DataFrame({
        "feature": df.columns,
        "special_ratio": df.isin(special_values).sum().values / df.shape[0]
    })

    zero_df = (
        (df == 0).sum()
        .div(df.shape[0])
        .reset_index()
        .rename(columns={"index": "feature", 0: "zero_ratio"})
    )

    null_df = (
        (df.isnull().sum() / df.shape[0])
        .to_frame("null_ratio")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    
    single_value_ratio = (
        df.apply(lambda x: x.value_counts(normalize=True, dropna=False).max())
        .reset_index()
        .rename(columns={"index": "feature", 0: "single_value_ratio"})
    )
    
    feature_quality_df = (
        special_df
        .merge(zero_df, on="feature", how="left")
        .merge(null_df, on="feature", how="left")
        .merge(single_value_ratio, on="feature", how="left")

    )

    feature_quality_df["max_bad_ratio"] = feature_quality_df[
        ["special_ratio", "zero_ratio", "null_ratio","single_value_ratio"]
    ].max(axis=1)

    feature_quality_df["drop_feature"] = (
        feature_quality_df["max_bad_ratio"] > max_threshold
        ) & ~feature_quality_df["feature"].str.contains("DPD|BAD", case=False)
    

    return feature_quality_df

def save_feature_quality_df(feature_quality_df: pd.DataFrame,feature_quality_csv_path:Path):
    """Persist feature quality report"""

    
    feature_quality_df.to_csv(feature_quality_csv_path, index=False)
    
    return feature_quality_csv_path


def load_clean_df(input_df: pd.DataFrame, feature_quality_df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad features based on quality report"""

    drop_features = feature_quality_df.loc[
        feature_quality_df["drop_feature"] == True, "feature"
    ].tolist()

    return input_df.drop(columns=drop_features, errors="ignore")

def orchastrator(df):
    """
    Orchestrates feature quality check:
    - loads transformed data
    - generates feature quality dataframe
    - saves the report
    - returns only the quality report file path
    """
    # read yaml
    params_path = FeatureQualityFilterConfig().params_path
    feature_quality_artifact = FeatureQualityFilterArtifact()

    params = read_yaml_file(params_path,logger)
    special_values = params['feature_quality_filter']['SPECIAL_VALUES']
    max_threshold = params['feature_quality_filter']['MAX_BAD_RATIO_THRESHOLD']



    feature_quality_df = generate_feature_quality_df(df,special_values,max_threshold)
    

    feature_quality_csv_path = feature_quality_artifact.feature_quality_save_path
    feature_quality_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    feature_quality_csv_path = save_feature_quality_df(feature_quality_df,feature_quality_csv_path)
    
    return feature_quality_csv_path