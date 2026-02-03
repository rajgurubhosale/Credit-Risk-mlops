import os
import pandas as pd
from src.constants.data_quality_check_constant import *
from src.entity.quality_check_artifact import *
from src.constants.data_transformation_constant import *
from src.entity.data_transformation_artifact import *

# constants
special_values = DataQualityConfig().special_values
max_threshold = DataQualityConfig().threshold


def generate_feature_quality_df(df: pd.DataFrame) -> pd.DataFrame:
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

    feature_quality_df = (
        special_df
        .merge(zero_df, on="feature", how="left")
        .merge(null_df, on="feature", how="left")
    )

    feature_quality_df["max_bad_ratio"] = feature_quality_df[
        ["special_ratio", "zero_ratio", "null_ratio"]
    ].max(axis=1)

    feature_quality_df["drop_feature"] = (
        feature_quality_df["max_bad_ratio"] > max_threshold
    )

    return feature_quality_df


def save_feature_quality_df(feature_quality_df: pd.DataFrame):
    """Persist feature quality report"""

    artifact_dir = ARTIFACT_DIR
    data_quality_dir = DataQualityConfig().data_quality_dir
    path = os.path.join(artifact_dir, data_quality_dir)

    os.makedirs(path, exist_ok=True)

    csv_path = os.path.join(path, "main_df_feature_quality_df.csv")
    feature_quality_df.to_csv(csv_path, index=False)

    return csv_path


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

    feature_quality_df = generate_feature_quality_df(df)

    feature_quality_csv_path = save_feature_quality_df(feature_quality_df)
    return feature_quality_csv_path