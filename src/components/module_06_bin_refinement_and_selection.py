import numpy as np
import pandas as pd
import pickle
import json
import gc
from src.constants.manual_bin_merging import *
from src.entity.manual_bin_merging_artifact import *
from src.components.module_05_woe_binning_and_selection import FeatureSelector
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.logger import config_logger
logger = config_logger('module_06_post_binning_feature_processing.py')

class PostBinManualBinMerger:
    def __init__(self):
        self.num_bins = {}
        
    def merge_bins_and_create_mapping(
        self,
        feature: str,
        bins_index: list,
        num_bins: dict
    ):

        bins_index = sorted(bins_index)
        df = num_bins[feature]
        df_calc = df.copy()
        bin_filter = df_calc.index.isin(bins_index)

        count_bin = df_calc.loc[bin_filter, 'Count'].sum()
        count_non_event = df_calc.loc[bin_filter, 'Non-event'].sum()
        count_event = df_calc.loc[bin_filter, 'Event'].sum()

        total_event = df_calc['Event'].sum()
        total_non_event = df_calc['Non-event'].sum()

        dist_non_event = count_non_event / total_non_event
        dist_event = count_event / total_event

        merged_woe = round(np.log(dist_non_event / dist_event), 6)
        
        old_woe_values = (
            pd.to_numeric(df_calc.loc[bin_filter, 'WoE'], errors='coerce')
            .round(6)
            .dropna()
            .unique()
            .tolist()
        )

        mapping = {old: merged_woe for old in old_woe_values}

        lower = df.loc[bins_index[0], 'Bin'].split(',')[0].replace('(', '')
        upper = df.loc[bins_index[-1], 'Bin'].split(',')[1].replace(')', '').strip()
        new_bin = f'({lower}, {upper}]'

        df.drop(index=bins_index[1:], inplace=True)

        total_count = df_calc.loc[df_calc.index != 'Totals', 'Count'].sum()
        count_pct = count_bin / total_count

        updates = {
            'Bin': new_bin,
            'Count': count_bin,
            'Count (%)': count_pct,
            'Non-event': count_non_event,
            'Event': count_event,
            'Event rate': count_event / count_bin,
            'WoE': merged_woe,
            'feature': feature
        }

        keep_idx = bins_index[0]
        df.loc[keep_idx, updates.keys()] = updates.values()

        return mapping
    
    def run_merge_plan_and_collect_mappings(
        self,
        feature: str,
        merge_plan: list,
        num_bins: dict
        ):
        """
        Returns:
        woe_mapping = {feature: {old_woe: new_woe}}
        """

        feature_mapping = {}

        for bins_index in merge_plan:
            mapping = self.merge_bins_and_create_mapping(
                feature=feature,
                bins_index=bins_index,
                num_bins=num_bins
            )
            feature_mapping.update(mapping)

        # --- recompute IV after all merges ---
        df = num_bins[feature].copy()   
        df_valid = df[df.index != 'Totals'].copy()

        total_event = df_valid['Event'].sum()
        total_non_event = df_valid['Non-event'].sum()

        df_valid['dist_event'] = df_valid['Event'] / total_event
        df_valid['dist_non_event'] = df_valid['Non-event'] / total_non_event

        df_valid['IV'] = (
            (df_valid['dist_non_event'] - df_valid['dist_event']) *
            df_valid['WoE']
        )

        df.loc[df_valid.index, 'IV'] = df_valid['IV'].values  
        # write back clean dataframe
        num_bins[feature] = df

        return {feature : feature_mapping}

    
    def run_all_merge_plans(
        self,
        merge_plans,
        num_bins
        ):
        """
        Execute merge plans for all specified features and accumulate WoE
        mappings in `self.all_woe_mapping`.
        Parameters
        ----------
        merge_plans : 
            Keys are feature names; values are merge plans
            Example::
                merge_plans = {
                    "loan_to_value": [[0, 1], [3, 4]],
                    "credit_score":  [[5, 6, 7]],
                }
        Returns
            Combined WoE mapping for every feature that was merged.
        """
        self.all_woe_mapping = {}

        for feature, merge_plan in merge_plans.items():
            if feature in num_bins:
                result = self.run_merge_plan_and_collect_mappings(
                    feature,
                    merge_plan,
                    num_bins
                )
                self.all_woe_mapping.update(result)

        return self.all_woe_mapping
 
    def apply_woe_remap(
        self,
        X_woe_df,
        woe_mapping,
        selected_features
    ) -> pd.DataFrame:
        """
        Remap old WoE values to merged WoE values.

        Parameters
        ----------
        X_woe_df : pd.DataFrame
            WoE-encoded dataframe (train / test / val)

        woe_mapping : dict
            {
                feature_name: {old_woe_value: new_woe_value}
            }
            
        selected_features : list
            Final selected features after feature selection
        Returns
        -------
        pd.DataFrame
            WoE-remapped dataframe
        """

        X_out = X_woe_df[selected_features].copy()
        
        # Only features that actually have WoE mappings
        woe_features = set(selected_features).intersection(woe_mapping.keys())

        for feature in woe_features:
            if feature in X_out.columns:
                X_out[feature] = X_out[feature].round(6)
                X_out[feature] = (
                    X_out[feature]
                    .replace(woe_mapping[feature])
                    .astype(float)
                )
        return X_out
    
    def get_num_bins(self):
        ''' load the num bins'''      
        
        # numerical feature bins
        with open(r'D:\home loan credit risk\artifact\binning\prebin\numerical_feature_bins.pkl', 'rb') as f:
            feature_bins_num = pickle.load(f)
   
  
        feature_list_path = r'D:\home loan credit risk\artifact\binning\prebin\selected_features.json'
        with open(feature_list_path, "r") as f:
            features = json.load(f)
            
        num_bins = {}
        for key,value in feature_bins_num.items():
            if key in features:
                num_bins[key] = value        
        del feature_bins_num,features
        gc.collect()
        
        return num_bins
    
    
class PostBinFeatureSelector:
    def __init__(self):
        self.iv_df = []
    
    def correlation_filter(self,X,final_features,iv_df):
        temp_df = iv_df[iv_df['feature'].isin(final_features)].copy()
        fe = FeatureSelector()
        selected_features  = fe.correlation_filter(X,final_features,temp_df)
        return selected_features
    
    def remove_multicollinearity_vif(
        self,
        X_woe: pd.DataFrame,
        iv_df: pd.DataFrame,
        threshold: float = 5.0,
        ) -> tuple[list[str], list[str]]:
        """
        Remove multicollinear features using VIF, dropping lowest-IV violator each round.

        Parameters
        ----------
        X_woe     : WOE-transformed feature matrix
        iv_df     : DataFrame with columns ['feature', 'IV']
        threshold : VIF threshold (default 5.0)

        Returns
        -------
        selected_features : list[str]
        dropped_features  : list[str]
        """
        X = X_woe.loc[:, X_woe.nunique() > 1].fillna(0).copy()

        iv = iv_df.set_index("feature")["IV"].to_dict()
        dropped = []

        
        while X.shape[1] > 1:
            mat = X.to_numpy()

            vif = np.array([variance_inflation_factor(mat, i) for i in range(mat.shape[1])])

            if vif.max() <= threshold:
                break

            # Among violators → drop the one with lowest IV
            mask      = vif > threshold
            violators = X.columns[mask]
            drop_col  = min(violators, key=lambda c: iv.get(c, 0.0))

            logger.info(f"Dropping '{drop_col}' (VIF={vif[X.columns.get_loc(drop_col)]:.2f}, IV={iv.get(drop_col, 0.0):.4f})")

            X = X.drop(columns=drop_col)
            dropped.append(drop_col)

        return X.columns.tolist(), dropped
    
# =========================================================
# Orchestrator
# =========================================================
class BinRefinementOrchestrator:

    def __init__(self):
        
        self.selected_features = []
        self.bin_merger = PostBinManualBinMerger()
        self.feature_selector = PostBinFeatureSelector()
        self.manualbinconfig = PostBinMergingConfig()
        
    def _save_final_dataframes(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Save final train and test DataFrames to CSV files in the specified artifact directory.

        Parameters
        ----------
        X_train : pd.DataFrame
            Final training dataset
        X_test : pd.DataFrame
            Final testing dataset
        artifact_dir : str
            Directory path to save the files
        """
        import os
        artifact_final_dir = self.manualbinconfig.artifact_final_dir
        
        os.makedirs(artifact_final_dir, exist_ok=True)

        # Define file paths
        train_path = os.path.join(artifact_final_dir, 'X_train_final.csv')
        test_path = os.path.join(artifact_final_dir, 'X_test_final.csv')

        # Save DataFrames
        X_train.to_csv(train_path, index=False)
        X_test.to_csv(test_path, index=False)
        
        logger.info(f"Final X_train saved at: {train_path}")
        logger.info(f"Final X_test saved at: {test_path}")
        
    def run(self):
        
        logger.info('Loading datasets')
        X_train = pd.read_csv(r'D:\home loan credit risk\artifact\binning\prebin\X_train_selected_woe.csv')
        X_test = pd.read_csv(r'D:\home loan credit risk\artifact\binning\prebin\X_test_selected_woe.csv')
        logger.info(f"Loaded X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        logger.info('RUNNING: MANUAL BIN MERGING')
        self.bin_merger.num_bins = self.bin_merger.get_num_bins()
        with open(r'D:\home loan credit risk\artifact\binning\post_bin_manual\selected_features.json',"r") as f:
            self.selected_features = json.load(f)
            
        merge_plans = self.manualbinconfig.bin_merging_plans
        woe_mapping = self.bin_merger.run_all_merge_plans(merge_plans,self.bin_merger.num_bins)
        logger.info('COMPLETED: MANUAL BIN MERGING')
        
        logger.info('Apply the Woe Mapping to features')
        X_train= self.bin_merger.apply_woe_remap(X_train,woe_mapping,self.selected_features)
        X_test = self.bin_merger.apply_woe_remap(X_test,woe_mapping,self.selected_features)
        logger.warning(f'SHAPE AFTER WOE MAPL X_TRAIN: {X_train.shape}, X_test:{X_test.shape}')
        
        logger.info('Correlation filtering started')
        
        iv_df = pd.read_csv(r'D:\home loan credit risk\artifact\binning\prebin\iv_df.csv')
        logger.info('iv df loaded successfully')
        
        self.selected_features = self.feature_selector.correlation_filter(X_train,self.selected_features,iv_df)
        
        # APPLY FEATURE SELECTION
        X_train = X_train[self.selected_features]
        X_test = X_test[self.selected_features]
        
        logger.info(f"Datasets Shape after the correlation feature selection - X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        final_features, dropped_features = self.feature_selector.remove_multicollinearity_vif(
        X_train,
        iv_df,
        threshold=5
        )

        logger.info(f"Selected {len(final_features)} features after VIF filtering")
        logger.info(f"Dropped features due to multicollinearity: {dropped_features}")
        
        X_train = X_train[final_features]
        X_test = X_test[final_features]
               
        logger.info('Pipeline completed')
        self._save_final_dataframes(X_train,X_test)
        
if __name__ =='__main__':
    obj = BinRefinementOrchestrator()
    obj.run()