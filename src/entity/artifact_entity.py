from pathlib import Path
from src.constants.artifacts_paths import *
from src.constants.bin_refinement_manual_constant import BIN_MERGING_PLANS
from dataclasses import dataclass,field


# WHAT DOES IT OUTPUTS
@dataclass
class DataIngestionArtifact:
    artifact_dir:Path = ARTIFACT_DIR
    artifact_data_raw_dir:Path = ARTIFACT_DATA_RAW_DIR
    application_data:Path = APPLICATION_DATA_PATH 
    bureau_data: Path = BUREAU_DATA_PATH 
    bureau_balance:Path = BUREAU_BALANCE_DATA_PATH
    credit_card_balance:Path = CREDIT_CARD_BALANCE_PATH
    installment_payments_data:Path = INSTALLMENT_PAYMENTS_DATA_PATH
    pos_cash_data:Path = POS_CASH_DATA_PATH
    previous_application_data:Path = PREVIOUS_APPLICATION_DATA_PATH
    
    
@dataclass
class DataValidationArtifact:
    #data_validation_yaml: dict = field(default_factory=lambda: DATA_VALIDATION_YAML)
    validation_report_path:Path = VALIDATION_REPORT_PATH
    
@dataclass
class DataTransformationArtifact:
    artifact_data_interim_dir:Path = ARTIFACT_DATA_INTERIM_DIR
    artifact_interim_main_transformed_df:Path = ARTIFACT_DATA_INTERIM_MAIN_DF_DIR
    
    

@dataclass
class FeatureQualityFilterArtifact:
    feature_quality_save_path :Path = VALIDATION_FEATURE_QUALITY_PATH
    
@dataclass
class FeatureEngArtifact:
    artifact_data_splits_dir:Path = ARTIFACT_DATA_SPLITS_DIR
    artifact_binning_dir:Path = ARTIFACT_BINNING_DIR
    artifact_bin_prebin_dir:Path = ARTIFACT_BINNING_PREBIN_DIR
    
    selected_x_train_path:Path = PREBIN_SELECTED_X_TRAIN_PATH
    selected_x_test_path:Path = PREBIN_SELECTED_X_TEST_PATH
    
    iv_df_path:Path = PREBIN_IV_DF_PATH
    cat_bin_df_path:Path = PREBIN_CAT_BIN_DF_PATH
    numerical_feature_bins_path:Path = PREBIN_NUMERICAL_FEATURE_BINS_PATHS
    selected_features_path:Path = PREBIN_SELECTED_FEATURES_PATHS
    
    data_splits_x_train_path:Path = DATA_SPLITS_X_TRAIN_PATH
    data_splits_x_test_path:Path =  DATA_SPLITS_X_TEST_PATH
    data_splits_y_train_path:Path = DATA_SPLITS_Y_TRAIN_PATH
    data_splits_y_test_path:Path =   DATA_SPLITS_Y_TEST_PATH
        
      
@dataclass
class FeatureBinMergingArtifact:
    artifact_data_final_dir:Path = ARTIFACT_DATA_FINAL_DIR
    artifact_postbin_manual_dir:Path = ARTIFACT_BINNING_POSTBIN_MANUAL_DIR  
    selected_features_path:Path = ARTIFACT_POSTBIN_SELECTED_FEATURES_PATH
    selected_num_bins_path:Path = ARTIFACT_POSTBIN_SELECTED_NUM_BINS_PATH
    X_train_final_path:Path = DATA_FINAL_X_TRAIN_PATH
    X_test_final_path:Path =  DATA_FINAL_X_TEST_PATH

    final_selected_features_path:Path = FINAL_SELECTED_FEATURES_PATH
    final_selected_num_bins_path:Path = FINAL_SELECTED_NUM_BINS_PATH
    final_model_features:Path = FINAL_MODEL_FEATURES
            
      
@dataclass
class ModelTrainigArtifact:
    artifact_model_dir:Path = ARTIFACT_MODEL_DIR
    model_path:Path = ARTIFACT_MODEL_PATH
    feature_importance_path: Path = MODEL_FEATURE_IMPORTANCE_PATH  # add this


@dataclass
class ModelEvalArtifact:
    metrics_path:Path = MODEL_EVAL_METRICS_PATH
    model_eval_dir:Path = ARTIFACT_MODEL_EVAL_DIR
    roc_curve_path:Path = MODEL_EVAL_ROC_CURVE_PATH


@dataclass
class ScorecardArtifact:
    scorecard_dir:Path = ARTIFACT_SCORECARD_DIR
    train_score_df_path:Path =  FINAL_SCORECARD_TRAIN_PATH

    scorecard_scaling_params_path:Path = SCORECARD_SCALING_PARAMS_PATH

    
    final_scorecard_table_path:Path = FINAL_SCORECARD_PERFORMANCE_TABLE_PATH
    scorecard_metrics_path:Path = SCORECARD_METRICS_PATH
    
    final_scorecard_rules_path:Path = FINAL_SCORECARD_RULES_PATH
    scorecard_categorical_rules:Path = SCORECARD_CATEGORICAL_RULES
    scorecard_numerical_rules:Path =SCORECARD_NUMERICAL_RULES
    scorecard_categorical_lookup:Path = SCORECARD_CATEGORICAL_LOOKUP
    scorecard_numerical_lookup:Path = SCORECARD_NUMERICAL_LOOKUP