import numpy as np
import pandas as pd
import pickle
import json
import gc
from src.utils.main_utils import downcast_df_variables,read_yaml_file
from src.constants.bin_refinement_manual_constant import *
from src.constants.artifacts_paths import *
from src.entity.artifact_entity import FeatureBinMergingArtifact,FeatureEngArtifact
from src.components.module_05_woe_binning_and_selection import WOEFeatureSelector
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.logger import config_logger
from src.entity.config_entity import FeatureBinMergingConfig

logger = config_logger('module_06_post_binning_feature_processing.py')

class PostBinManualBinMerger:
    ''' This class loads the bin merge plans from postbin_manual dir
        that is created manually in notebook manual bins merging  and
        returns the the updated bined data and the num_bins summay plans
        '''
    def __init__(self):
        self.num_bins = {}
        
    def _compute_merged_woe_for_bin_group(
        self,
        feature: str,
        bins_index: list,
        num_bins: dict
        ):
        ''' 
        run the bin merge plan and return the woe mapping for single group:
        return:
            woe_mapping:
            feature:{
                old_woe:new_woe,
                old_woe:new_woe,
            }
        
        '''

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
        upper = df.loc[bins_index[-1], 'Bin'].split(',')[1].replace(')', '').replace(']', '').strip()
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
            mapping = self._compute_merged_woe_for_bin_group(
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
        logger.info(f'all Feature Mapping created suceesfully')

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
        
    


    
class PostBinFeatureSelector:
    def __init__(self):
        self.iv_df = []
    
    def correlation_filter(self,X,final_features,iv_df,corr_threshold):
        temp_df = iv_df[iv_df['feature'].isin(final_features)].copy()
        fe = WOEFeatureSelector(corr_threshold=corr_threshold)
        selected_features  = fe.correlation_filter(X,final_features,temp_df)
        return selected_features
    
    def multicollinearity_vif_filter(
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
        X = X.astype(np.float32)  
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
    
    def psi_filter(self,X_train_woe, X_test_woe, psi_threshold=0.25, eps=1e-6):
        """
        Perform Population Stability Index (PSI) check between
        training and testing datasets for WoE-transformed features.

        The Home Credit Kaggle dataset does not contain time stamps
        so out-of-time (OOT) sample cannot be constructed,
        and population stability analysis (PSI) is not meaningful in a temporal sense.
        there for the testing data is used as the to check psi
        (it will not remove any feature because them have the same distributions)
        but it is implemented for credit risk standard pipline purpose

        """

        kept_features = []
        removed_features = []

        for feature in X_train_woe.columns:
            # Distribution
            train_dist = X_train_woe[feature].value_counts(normalize=True)
            test_dist  = X_test_woe[feature].value_counts(normalize=True)

            # Align bins
            all_bins = train_dist.index.union(test_dist.index)
            train_dist = train_dist.reindex(all_bins, fill_value=0) + eps
            test_dist  = test_dist.reindex(all_bins, fill_value=0) + eps

            # PSI formula
            psi = np.sum((train_dist - test_dist) * np.log(train_dist / test_dist))

            # Decision
            if psi > psi_threshold:
                logger.info(f" REMOVE | {feature} | PSI = {psi:.4f}")
                removed_features.append(feature)
            else:
                logger.info(f" KEEP   | {feature} | PSI = {psi:.4f}")
                kept_features.append(feature)

        return kept_features, removed_features

class BinRefinementorchestrate:

    def __init__(self):
        
        self.selected_features = []

        self.manual_bin_artifact = FeatureBinMergingArtifact()
        self.manual_bin_config = FeatureBinMergingConfig()

        self.bin_merger = PostBinManualBinMerger()
        self.feature_selector = PostBinFeatureSelector()
        self.fe_artifact = FeatureEngArtifact()

    def get_num_bins(self):
        ''' load the num bins selected features only'''      
        num_bins_path = self.fe_artifact.numerical_feature_bins_path
        
        # numerical feature bins
        with open(num_bins_path, 'rb') as f:
            feature_bins_num = pickle.load(f)
   
        
        feature_list_path = self.manual_bin_artifact.selected_features_path
        with open(feature_list_path, "r") as f:
            features = json.load(f)
            
        num_bins = {}
        for key,value in feature_bins_num.items():
            if key in features:
                num_bins[key] = value        
                
        del feature_bins_num,features        
        gc.collect()
        
        return num_bins
    def save_final_features_bins(self,num_bins,final_selected_features):
        
        ''' save the final filtered feature for model and
            num bins in the final dir'''
        feature_path = self.manual_bin_artifact.final_selected_features_path 
        feature_path.parent.mkdir(parents=True,exist_ok=True)
        
        with open(feature_path, "w") as f:
            json.dump(final_selected_features, f, indent=4)
            print(feature_path)    
            
            
        final_selected_num_bins_path = self.manual_bin_artifact.final_selected_num_bins_path
        final_selected_num_bins_path.parent.mkdir(parents=True,exist_ok=True)
        
        final_num_bins = {}
        for key,value in num_bins.items():
            if key in final_selected_features:
                final_num_bins[key] = value     
                
                
        with open(final_selected_num_bins_path, "wb") as f:
            pickle.dump(final_num_bins, f)
        logger.info(f'Final selected feature and num bins are saved in {self.manual_bin_artifact.artifact_data_final_dir}')
    
        
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
        
        train_path = self.manual_bin_artifact.X_train_final_path
        test_path = self.manual_bin_artifact.X_test_final_path

        # Save DataFrames
        X_train.to_csv(train_path, index=False)
        X_test.to_csv(test_path, index=False)
        
        logger.info(f"Final X_train saved at: {train_path}")
        logger.info(f"Final X_test saved at: {test_path}")
        
    def run(self):
        
        logger.info('Loading datasets')
        X_train_selected_path = self.fe_artifact.selected_x_train_path
        
        params = read_yaml_file(PARAMS_DIR_PATH,logger)
        self.params = params.get('bin_refinement')

        X_train = pd.read_csv(X_train_selected_path)
        #X_test = pd.read_csv(r'D:\home loan credit risk\artifact\binning\prebin\X_test_selected_woe.csv')
        X_train = downcast_df_variables(X_train)
        
        logger.info(f"Loaded X_train: {X_train.shape}")
        
        logger.info('RUNNING: MANUAL BIN MERGING')
            
        num_bins = self.get_num_bins()
    
        with open(self.manual_bin_artifact.selected_features_path,"r") as f:
            selected_features = json.load(f)
        
        
        merge_plans = self.manual_bin_config.bin_merging_plans
        woe_mapping = self.bin_merger.run_all_merge_plans(merge_plans,num_bins)
        logger.info('COMPLETED: MANUAL BIN MERGING')
        
        logger.info('Apply the Woe Mapping to features')
        X_train= self.bin_merger.apply_woe_remap(X_train,woe_mapping,selected_features)
        logger.info(f'WoE Mapping applied on X_train: {X_train.shape}')
        
        X_test_path = self.fe_artifact.selected_x_test_path
        X_test_path.parent.mkdir(parents=True,exist_ok=True)
        
        X_test = pd.read_csv(X_test_path)
        X_test = downcast_df_variables(X_test)

        X_test = self.bin_merger.apply_woe_remap(X_test,woe_mapping,selected_features)
        logger.info(f'WoE Mapping applied on X_test: {X_test.shape}')
        
        logger.info('Correlation filtering started')
        iv_df = pd.read_csv(self.fe_artifact.iv_df_path)
        logger.info('iv df loaded successfully')
        
        correlation_threshold = self.params['correlation_threshold']
        corr_selected_features = self.feature_selector.correlation_filter(X_train,selected_features,iv_df,correlation_threshold)        

        # APPLY FEATURE SELECTION
        X_train = X_train[corr_selected_features].copy()
        X_test = X_test[corr_selected_features].copy()
        logger.info('Correlation Feature selection completed')
        logger.info(f"Datasets Shape after the correlation feature selection - X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        vif_threshold = self.params['vif_threshold']
        vif_features, dropped_features = self.feature_selector.multicollinearity_vif_filter(
            X_train, iv_df, vif_threshold
        )
        #dropped_features = ['PA_AVG_AMT_ANNUITY_CARDS', 'CB_STD_PAYMENT_VOLATILITY_9M', 'IP_DPD_TREND', 'CB_MAX_RATIO_AMT_PAYMENT_MIN_INST_9M', 'CB_MAX_RATIO_PAYMENT_BALANCE_24M', 'PA_RATIO_HC_REFUSED_LOANS', 'CB_MAX_RATIO_PAYMENT_BALANCE_3M', 'PA_LOANS_REFUSED_RECENT_1080D', 'B_CLOSED_CREDIT_RATIO']
        #vif_features = [f for f in corr_selected_features if f not in dropped_features] 

                

        logger.info(f"Selected {len(vif_features)} features after VIF filtering")
        logger.info(f"Dropped due to multicollinearity: {dropped_features}")

        
        X_train = X_train[vif_features].copy()
        X_test = X_test[vif_features].copy()
        # ── PSI filter ───────────────────────────────────────────────────────
        logger.info('Running PSI stability check')
        psi_threshold = self.params['psi_threshold']
        
        psi_features, psi_removed = self.feature_selector.psi_filter(
            X_train, X_test, psi_threshold)
        logger.info(f"Selected {len(psi_features)} features after PSI check")
        logger.info(f"Removed due to high PSI: {psi_removed}")
        logger.info(f'len PSi features:{psi_features}')
        X_train = X_train[psi_features].copy()
        X_test  = X_test[psi_features].copy()
 
        final_features = psi_features      
        
        logger.info('Bin refindement pipeline completed')
        self._save_final_dataframes(X_train,X_test)
        self.save_final_features_bins(num_bins,final_features)
        
        
        
if __name__ =='__main__':
    orchestrate = BinRefinementorchestrate()
    orchestrate.run()