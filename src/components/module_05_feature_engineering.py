from src.entity.data_transformation_artifact import *
from src.constants.data_transformation_constant import *
from src.entity.feature_engineering_artifact import *
from src.constants.feature_engineering_constant import *
from src.components.module_04_feature_quality_check import orchastrator,load_clean_df
import json
from sklearn.model_selection import train_test_split
import pickle
import logging
from src.logger import config_logger
import os 
import sys
from src.exception import MyException
import numpy as np
import pandas as pd
from optbinning import OptimalBinning
import gc

logger = config_logger('module_05_feature_engineering.py')


id_col = "SK_ID_CURR"
target_col = "TARGET"

def handle_missing_values(df, id_col="SK_ID_CURR"):
    num_cols = (
        df.select_dtypes(include=[np.number])
        .select_dtypes(exclude=["bool"])
        .columns
        .drop(id_col)
    )

    df[num_cols] = df[num_cols].fillna(FeatureEngConfig().default_num_missing_value)
    return df


#-----------------------------------------------------------------
class NumericalWOEBinner:
    """
    Numerical WOE Binning using optbinning
    - Fit on TRAIN only
    - Transform TRAIN / TEST safely
    """
    def __init__(self, id_col, special_codes=None, max_n_bins=20, min_prebin_size=0.05,prebinning_method='quantile'):
                
        self.id_col = id_col
        self.special_codes = special_codes
        self.max_n_bins = max_n_bins
        self.min_prebin_size = min_prebin_size
        self.prebinning_method = prebinning_method
        self.binning_models = {}
        self.iv_df = None
        self.numerical_features = []

    def fit(self, X_train_num, y_train):
        """
        X_train_num: pd.DataFrame containing only numerical features including id_col
        """
        self.numerical_features = [c for c in X_train_num.columns if c != self.id_col]
        iv_records = []

        for col in self.numerical_features:
            try:
                optb = OptimalBinning(
                    name=col,
                    dtype="numerical",
                    solver="cp",
                    prebinning_method=self.prebinning_method,
                    min_prebin_size=self.min_prebin_size,
                    max_n_bins=self.max_n_bins,
                    special_codes=self.special_codes
                )

                # Fit on TRAIN only
                optb.fit(X_train_num[col], y_train)

                # Build binning table
                bt = optb.binning_table
                bt.build()

                # Store model
                self.binning_models[col] = optb

                # Store IV
                iv_records.append({
                    "feature": col,
                    "iv": bt.iv
                })

            except MyException as e:
                print(f"Skipped {col}: {e}")

        self.iv_df = pd.DataFrame(iv_records).sort_values(by="iv", ascending=False).reset_index(drop=True)
        return self

    def transform(self, X_num):
        X_woe = pd.DataFrame(index=X_num.index)
        X_woe[self.id_col] = X_num[self.id_col]

        for col, optb in self.binning_models.items():
            if col not in X_num.columns:
                continue
            X_woe[col + "_WOE"] = optb.transform(X_num[col].fillna(-99999), metric="woe")

        return X_woe

    def fit_transform(self, X_train_num, y_train):
        self.fit(X_train_num, y_train)
        return self.transform(X_train_num)

#-----------------------------------------------------------------
class CategoricalWOEBinner:
    """
    Categorical WOE Binning
    - One bin per category
    - Rare grouping
    - Train-only WOE calculation
    - Safe transform for TEST
    """
    def __init__(self, id_col, rare_threshold=0.01, eps=1e-6):
        self.id_col = id_col
        self.rare_threshold = rare_threshold
        self.eps = eps

        self.categorical_features = []
        self.cat_woe_maps = {}
        self.iv_df = None
        
    def get_categorical_woe_bins(self,cat_woe_maps):
        records = []

        for feature, info in cat_woe_maps.items():
            woe_map = info["woe_map"]
            rare_cats = info["rare_categories"]

            for bin_name, woe in woe_map.items():
                records.append({
                    "feature": feature,
                    "bin": bin_name,
                    "woe": woe,
                    "is_rare_bin": bin_name == "RARE",
                    "rare_categories": ", ".join(rare_cats) if bin_name == "RARE" else None
                })

        return pd.DataFrame(records)
    
    def fit(self, X_train_cat, y_train):
        """
        X_train_cat: pd.DataFrame containing only categorical features including id_col
        """
        self.categorical_features = [c for c in X_train_cat.columns if c != self.id_col]
        cat_iv_records = []
        
        for col in self.categorical_features:
            train_series = X_train_cat[col].fillna("MISSING")

            # Frequency on TRAIN
            freq = train_series.value_counts(normalize=True)


            # Rare categories
            rare_categories = freq[freq < self.rare_threshold].index.tolist()

            # Apply rare grouping
            train_binned = train_series.where(~train_series.isin(rare_categories), "RARE")

            df_tmp = pd.DataFrame({"cat": train_binned, "target": y_train})

            # Aggregate
            agg = df_tmp.groupby("cat").agg(total=("target", "count"), bad=("target", "sum")).reset_index()
            agg["good"] = agg["total"] - agg["bad"]

            total_good = agg["good"].sum()
            total_bad = agg["bad"].sum()

            # Distributions
            agg["dist_good"] = agg["good"] / total_good
            agg["dist_bad"] = agg["bad"] / total_bad

            # WOE
            agg["woe"] = np.log((agg["dist_good"] + self.eps) / (agg["dist_bad"] + self.eps))

            # IV
            agg["iv"] = (agg["dist_good"] - agg["dist_bad"]) * agg["woe"]
            iv_value = agg["iv"].sum()

            # Store WOE map for this feature
            woe_map = dict(zip(agg["cat"], agg["woe"]))
            self.cat_woe_maps[col] = {"woe_map": woe_map, "rare_categories": rare_categories}

            # Store IV
            cat_iv_records.append({"feature": col, "iv": iv_value})

        self.iv_df = pd.DataFrame(cat_iv_records).sort_values(by="iv", ascending=False).reset_index(drop=True)
        return self

    def transform(self, X_cat):
        X_woe = pd.DataFrame(index=X_cat.index)
        X_woe[self.id_col] = X_cat[self.id_col]

        for col in self.categorical_features:
            if col not in self.cat_woe_maps:
                continue

            woe_map = self.cat_woe_maps[col]["woe_map"]
            rare_categories = self.cat_woe_maps[col]["rare_categories"]

            if "RARE" not in woe_map:
                woe_map["RARE"] = 0.0

            test_series = X_cat[col].fillna("MISSING")
            test_binned = test_series.where(~test_series.isin(rare_categories), "RARE")
            test_binned = test_binned.where(test_binned.isin(woe_map.keys()), "RARE")
            X_woe[col + "_WOE"] = test_binned.map(woe_map)

        return X_woe

    def fit_transform(self, X_train_cat, y_train):
        self.fit(X_train_cat, y_train)
        return self.transform(X_train_cat)
#-----------------------------------------------------------------

class FeatureSelector:
    def __init__(self,iv_threshold=0.02,corr_threshold=0.7,id_col=None):
        self.iv_threshold = iv_threshold
        self.corr_threshold = corr_threshold
        self.id_col = id_col
        self.feature_bins_num = {}
        self.removed_low_iv_ = None
        self.removed_corr_ = None
    
    def iv_filteration(self, X_woe, iv_df):
        """
        Remove features with IV < threshold 
        return:
            keep_features: list of the features that should be kept and those are Iv>threshold
        """
        try:
            if self.id_col in X_woe.columns:
                X_woe = X_woe.drop(columns=[self.id_col])
                
            iv_df["feature"] = iv_df["feature"] + "_WOE"

            # keep features above threshold
            filt = iv_df["iv"] >= self.iv_threshold
            
            keep_features= iv_df.loc[filt,"feature"].tolist()

            self.removed_low_iv_ = iv_df.loc[
                iv_df["iv"] < self.iv_threshold, ["feature", "iv"]
            ]
                        
            
            logger.info(f"Kept Feature {len(keep_features)} features with IV > {self.iv_threshold}")           
            
            logger.info(f"Removed {self.removed_low_iv_.shape[0]} features with IV < {self.iv_threshold}")           
            
        except Exception as e:
            raise MyException(e,sys,logger)
        
        return keep_features

    
    def calculate_num_feature_quality_scores(self,runner,X_train, keep_features, iv_df):
        """
        Calculate numerical features feature quality for WOE-transformed numerical features.
        
        Parameters:
        -----------
        runner : object
            The pipeline or object containing num_binner with WOE binning models.
        X_filtered_woe : pd.DataFrame
            DataFrame containing WOE-transformed features.
        iv_df : pd.DataFrame
            DataFrame containing IV values with columns ['feature', 'iv'].
        
        Returns:
        --------
        feature_quality : pd.DataFrame
            DataFrame with feature quality scores, missing rate, special rate, and IV.
        """
        
        X_filtered_woe = X_train[keep_features]
        
        # Build WOE binning tables for all numerical features
        try:
            for feature, optb in runner.num_binner.binning_models.items():
                bt = optb.binning_table
                bt_df = bt.build()
                bt_df["feature"] = feature
                self.feature_bins_num[feature] = bt_df

            # Extract missing and special value ratios
            rows = []
            for feature, temp in self.feature_bins_num.items():
                # Missing values
                missing_ratio = temp.loc[temp['Bin'] == 'MISSING', 'Count (%)'].sum()
                
                # Special values
                special_mask = temp['Bin'].isin(['SC_-99999', 'SC_-88888', 'SC_-77777'])
                special_ratio = temp.loc[special_mask, 'Count (%)'].sum()
                
                rows.append({
                    "feature": feature + "_WOE",  # add _WOE suffix
                    "missing_rate": missing_ratio,
                    "special_rate": special_ratio
                })

            feature_df = pd.DataFrame(rows)


            # Merge with IV values
            
            iv_series = (
                iv_df
                .set_index("feature")["iv"]
                .loc[X_filtered_woe.columns]
            )
            
            iv_series = iv_series.to_frame().reset_index()
            iv_series = iv_series.rename(columns={'index':'feature'})

            num_feature_quality = feature_df.merge(iv_series, on='feature', how='left')
            num_feature_quality = num_feature_quality[num_feature_quality['iv'].notnull()]

            # Compute score: iv - (missing + special)
            
            num_feature_quality['score'] =  (0.7 * num_feature_quality['iv']) - (
                (0.2 * num_feature_quality['missing_rate']) + (0.1 * num_feature_quality['special_rate'])
            )
            # Set index as feature
            num_feature_quality = num_feature_quality.set_index('feature')
            logger.info(f'Num quality feature scores head and tail')
            logger.info(num_feature_quality.head(5))
            logger.info(num_feature_quality.tail(5))
            
            
        except Exception as e:
            raise MyException(e,sys,logger)
 
        logger.info(f"Calculated numerical feature quality for {num_feature_quality.shape[0]} features")
        return num_feature_quality
   

    def calculate_cat_feature_quality_scores(self,runner, X_train_raw, keep_features, iv_cat):
        """
        Calculate feature quality for categorical features using WOE binner information.

        Parameters:
        -----------
        runner : object
            The pipeline or object containing cat_binner with categorical features and WOE maps.
        X_train_raw : pd.DataFrame
            Original training data (before WOE transformation).
        iv_cat : pd.DataFrame
            DataFrame containing IV values for categorical features with columns ['feature', 'iv'].

        Returns:
        --------
        cat_feature_quality : pd.DataFrame
            DataFrame with categorical feature quality scores, missing rate, rare rate, and IV.
        """
        feature_list_update= []
        for feature in keep_features:
            feature = feature.replace('_WOE','')
            feature_list_update.append(feature)
            
        rows = []
        try:
                        # Use the raw X_train (before WOE transformation)
            # Or if you saved the original X_train somewhere before WOE, use that
            for feature in runner.cat_binner.categorical_features:  # raw names
                total_count = len(X_train_raw)

                # Missing rate: check original raw column
                if feature in X_train_raw.columns:
                    missing_rate = X_train_raw[feature].isna().sum() / total_count
                    rare_categories = runner.cat_binner.cat_woe_maps[feature]["rare_categories"]
                    rare_rate = X_train_raw[feature].isin(rare_categories).sum() / total_count
                else:
                    # If the raw column is not available, fallback to 0
                    missing_rate = 0
                    rare_rate = 0

                
                # IV from binner
                iv_value = iv_cat.loc[
                    iv_cat["feature"] == feature, "iv"
                ].values[0]

                rows.append({
                    "feature": feature + "_WOE",
                    "missing_rate": missing_rate,
                    "rare_rate": rare_rate,
                    "iv": iv_value
                })
            cat_feature_quality = pd.DataFrame(rows)

            # Feature quality score
            cat_feature_quality["score"] = (
                (0.7 * cat_feature_quality["iv"])
                - (0.2 * cat_feature_quality["missing_rate"])
                - (0.1 * cat_feature_quality["rare_rate"])
            )
            cat_feature_quality = cat_feature_quality.set_index('feature')
            logger.info(f'cat quality feature scores head and tail')
            logger.info(cat_feature_quality.head(5))
            logger.info(cat_feature_quality.tail(5))
            
            
        except Exception as e:
            raise MyException(e,sys,logger)
        logger.info(f"Calculated categorical feature quality for {cat_feature_quality.shape[0]} features")
        return cat_feature_quality

    def _merge_num_cat_feature_quality(self,num_feature_quality,cat_feature_quality):
        ''' merge both numerical and categorical feature quality scores dataframe into feature quality final df'''
        try:
            
            feature_quality_score_final = pd.concat([num_feature_quality, cat_feature_quality], axis=0)
            feature_quality_score_final = feature_quality_score_final.sort_values("score", ascending=False)
        except Exception as e:
            raise MyException(e,sys,logger)
        
        logger.info(f"Merged numerical and categorical features: {feature_quality_score_final.shape[0]} total")
        logger.info(f'head and tail of final feature qulity score df ')
        
        logger.info(feature_quality_score_final.head(5))
        logger.info(feature_quality_score_final.tail(5))
        
        return feature_quality_score_final
    
    def correlation_filter(self,X_train_woe, keep_features, feature_quality_score_final):
        """
        Remove highly correlated features based on a threshold, keeping the feature with the highest score.

        Parameters:
        -----------
        X_filtered_woe : pd.DataFrame
            WOE-transformed features dataframe.
        feature_quality : pd.DataFrame
            DataFrame containing feature quality scores (index=feature, must contain 'score' column).
        corr_threshold : float, optional
            Correlation threshold above which one of the features will be dropped. 
            If None, uses self.corr_threshold.

        Returns:
        --------
        selected_features : list
            List of features selected after removing highly correlated features.
        dropped_features : list
            List of features that were dropped due to high correlation.
        """
                
        try:

            corr_threshold = self.corr_threshold
            X_filtered_woe = X_train_woe[keep_features]
            
            # Ensure alignment
            X_filtered_woe = X_filtered_woe.astype(np.float32)
            feature_quality_score_final = feature_quality_score_final.copy()

            # Keep only common features
            common_features = X_filtered_woe.columns.intersection(feature_quality_score_final.index)
            X_filtered_woe = X_filtered_woe[common_features]
            feature_quality_score_final = feature_quality_score_final.loc[common_features]

            # ==============================
            # CORRELATION MATRIX
            # ==============================
            corr_matrix = X_filtered_woe.corr().abs()

            # Upper triangle (avoid duplicates & self-corr)
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            drop_features = set()
            visited = set()

            # ==============================
            # CORR + SCORE FILTER LOGIC
            # ==============================
            for feature in upper.columns:

                if feature in visited or feature in drop_features:
                    continue

                # Find all features correlated with `feature`
                correlated = upper.index[upper[feature] > corr_threshold].tolist()

                if not correlated:
                    visited.add(feature)
                    continue

                # Form correlation group (feature + its correlated ones)
                group = set(correlated + [feature])

                # Remove already dropped features
                group = [f for f in group if f not in drop_features]

                if len(group) <= 1:
                    visited.update(group)
                    continue

                # Select best feature based on SCORE
                best_feature = (
                    feature_quality_score_final
                    .loc[group]
                    .sort_values("score", ascending=False)
                    .index[0]
                )

                # Drop all others
                for f in group:
                    if f != best_feature:
                        drop_features.add(f)

                visited.update(group)
                
                logger.debug(f"Corr group: {group} | kept: {best_feature}")

    
      
            selected_features = [f for f in X_filtered_woe.columns if f not in drop_features]
            self.removed_corr_ = list(drop_features)
            
            logger.info(f"Total features before : {X_filtered_woe.shape[1]}")
            logger.info(f"Total features after  : {len(selected_features)}")
            logger.info(f"Dropped features      : {len(self.removed_corr_)}")


        except Exception as e:
            raise MyException(e,sys,logger)

        logger.info(f"Removed {len(self.removed_corr_)} features due to high correlation (> {self.corr_threshold})")
        logger.info(f"Remaining feature list final shape is  {len(selected_features)}")
        
        return selected_features
    def fit(self, X_train_woe,X_train_raw, runner, iv_num, iv_cat):
        """
        Fit the selector:
        - filter by IV
        - compute numerical and categorical feature quality scores
        - merge scores
        - filter correlated features
        """
        try:
            # 1️⃣ IV filtering
            keep_features = self.iv_filteration(X_train_woe, iv_num)
            logger.info(f'IV filtering DONE ')
            
            # 2️⃣ Numerical features quality
            num_quality = self.calculate_num_feature_quality_scores(runner, X_train_woe, keep_features, iv_num)
            logger.info(f'Numerical features quality scores genrated succefully')

            # 3️⃣ Categorical features quality
            cat_quality = self.calculate_cat_feature_quality_scores(runner, X_train_raw, keep_features, iv_cat)
            logger.info(f'Categorical features quality scores genrated succefully')

            # 4️⃣ Merge
            feature_quality_score_final = self._merge_num_cat_feature_quality(num_quality, cat_quality)
            logger.info(f'feature quality combined succesfully')
            
            # 5️⃣ Correlation filtering
            self.selected_features_ = self.correlation_filter(X_train_woe,keep_features, feature_quality_score_final)
            logger.info(f'Correlation filtering DONE succesfully')

        except Exception as e:
            raise MyException(e, sys, logger)
        
        return self  # allow chaining like sklearn

    # -------------------------
    # TRANSFORM
    # -------------------------
    def transform(self, X):
        """
        Filter the dataframe to only selected features.
        """
        if self.selected_features_ is None:
            raise RuntimeError("FeatureSelector not fitted. Call `.fit()` first.")

        # Ensure only features present in X
        available_features = [f for f in self.selected_features_ if f in X.columns]
        missing_features = list(set(self.selected_features_) - set(available_features))

        if missing_features:
            logger.warning(f"{len(missing_features)} selected features missing in input, skipping them")
            logger.debug(f"Missing features: {missing_features}")

        return X[available_features]

    # -------------------------
    # FIT + TRANSFORM
    # -------------------------
    def fit_transform(self,X_train_woe,X_train_raw, runner, iv_num, iv_cat):
        self.fit(X_train_woe,X_train_raw, runner, iv_num, iv_cat)
        return self.transform(X_train_woe)

class FeatureEngineeringPipeline:
    def __init__(self, id_col, target_col):
        self.id_col = id_col
        self.target_col = target_col
        self.fe_config = FeatureEngConfig()
        self.be_config = BinningConfig()
        
        
        #config
        self.num_binner = NumericalWOEBinner(
            id_col=id_col,
            special_codes=self.fe_config.special_codes,
            max_n_bins= self.be_config.max_n_bins,
            min_prebin_size=self.be_config.min_bin_pct,
            prebinning_method = self.be_config.prebinning_method
        )

        self.cat_binner = CategoricalWOEBinner(
            id_col=id_col,
            rare_threshold=self.be_config.rare_threshold
        )
        
    
    def fit(self, X_train, y_train):

        logger.info("Fitting Feature Engineering (TRAIN only)")

        # -----------------------------
        # handle missing
        X_train = handle_missing_values(X_train, self.id_col)

        # identify columns
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        num_cols = list(dict.fromkeys([self.id_col] + num_cols))
        cat_cols = list(dict.fromkeys([self.id_col] + cat_cols))
        
   
   
        X_train_num = X_train[num_cols]
        X_train_cat = X_train[cat_cols]

        # -----------------------------
        # FIT + TRANSFORM (TRAIN)
        logger.info("Fitting & transforming Numerical WOE (TRAIN)")
        self.X_train_num_woe_ = (
            self.num_binner.fit_transform(X_train_num, y_train)
            .astype(np.float32)
        )
        

        logger.info("Fitting & transforming Categorical WOE (TRAIN)")
        self.X_train_cat_woe_ = (
            self.cat_binner.fit_transform(X_train_cat, y_train)
            .astype(np.float32)
        )
        
        del X_train_num,X_train_cat
        # -----------------------------
        # store IVs
        self.iv_num_ = self.num_binner.iv_df.copy()
        self.iv_cat_ = self.cat_binner.iv_df.copy()

        self.iv_df_ = (
            pd.concat([self.iv_num_, self.iv_cat_], axis=0)
            .reset_index(drop=True)
        )

        logger.info("Feature Engineering FIT completed")

        return self
    def transform(self, X):

        logger.info("Transforming data using trained WOE")

        X = handle_missing_values(X, self.id_col)

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        num_cols = list(dict.fromkeys([self.id_col] + num_cols))
        cat_cols = list(dict.fromkeys([self.id_col] + cat_cols))

        X_num = X[num_cols]
        X_cat = X[cat_cols]

        X_num_woe = self.num_binner.transform(X_num)
        X_cat_woe = self.cat_binner.transform(X_cat)
        
      
        X_final_woe = pd.concat(
            [
                X_num_woe.drop(columns=[self.id_col]),
                X_cat_woe.drop(columns=[self.id_col])
            ],
            axis=1
        ).astype(np.float32)

        logger.info(f"Transformation completed | Shape: {X_final_woe.shape}")
        del num_cols,cat_cols
        
        return X_final_woe
    
    def fit_transform(self, df):

        logger.info("Starting Feature Engineering FIT_TRANSFORM")

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        #config
        random_state = self.fe_config.default_random_state
        test_size = self.fe_config.test_size
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # save the dataset into memory for after work
        #config
        splits_dir_path = self.fe_config.splits_dir
        
        os.makedirs(splits_dir_path,exist_ok=True)

        X_train_path = os.path.join(splits_dir_path,'X_train.csv')
        X_test_path =  os.path.join(splits_dir_path,'X_test.csv')
        y_train_path =  os.path.join(splits_dir_path,'y_train.csv')
        y_test_path =  os.path.join(splits_dir_path,'y_test.csv')
        
        X_train.to_csv(X_train_path,index=False)
        X_test.to_csv(X_test_path,index=False)
        y_train.to_csv(y_train_path,index=False)
        y_test.to_csv(y_test_path,index=False)
        
                
        # delete the Df  and the X to free up memory 
        del  X, df
        gc.collect()
        
        
        
        # FIT (creates TRAIN WOE dfs)
        self.fit(X_train, y_train)

        # MERGE TRAIN WOE
        X_train_final_woe = pd.concat(
            [
                self.X_train_num_woe_.drop(columns=[self.id_col]),
                self.X_train_cat_woe_.drop(columns=[self.id_col])
            ],
            axis=1
        )
        
        # to save memory
        self.X_train_cat_woe_ = None
        self.X_train_cat_woe_ = None
        gc.collect()
        
        # TRANSFORM TEST
        X_test_final_woe = self.transform(X_test)
 
        logger.info("Feature Engineering FIT_TRANSFORM completed")
        # categorical bins dataset
        cat_woe_maps = self.cat_binner.cat_woe_maps
        cat_bin_df = self.cat_binner.get_categorical_woe_bins(cat_woe_maps)
        
        

        return {
            'cat_bin_df':cat_bin_df,
            'X_train_df':X_train,
            "X_train_final_woe": X_train_final_woe,
            "X_test_final_woe": X_test_final_woe,
            "y_train": y_train,
            "y_test": y_test,

            # diagnostics
            "iv_df": self.iv_df_,
            "iv_num_train": self.iv_num_,
            "iv_cat_train": self.iv_cat_,
        }


if __name__ == "__main__":
    
    runner = FeatureEngineeringPipeline(id_col, target_col)
    
    selector = FeatureSelector(iv_threshold=0.02, corr_threshold=0.7, id_col=id_col)


    try:
        logger.info("Pipeline execution started")


        data_path = r"D:\home loan credit risk\artifact\interim\main_df_transformed.csv"
        logger.info(f"Loading dataset from: {data_path}")
       
        df = pd.read_csv(data_path)
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        logger.info(f'Main df shape:{df.shape} ')
        
        # module 4 plugins
        feature_quality_csv_path = orchastrator(df)
        feature_quality_df = pd.read_csv(feature_quality_csv_path)
        

        df = load_clean_df(df, feature_quality_df)

        data_quality_df = pd.read_csv(r'D:\home loan credit risk\artifact\data_quality\main_df_feature_quality_df.csv')

        logger.info(f'df shape after removing feature with more> 95 % of nan or special values:{df.shape} ')
        
        artifacts = runner.fit_transform(df)

         
        X_train_raw = artifacts['X_train_df']
        X_train = artifacts["X_train_final_woe"]
        X_test  = artifacts["X_test_final_woe"]
        cat_bin_df = artifacts['cat_bin_df']
        iv_num = artifacts["iv_num_train"]
        iv_cat = artifacts["iv_cat_train"]
        
        #iv_df the WOE in front of the features
        iv_df = artifacts['iv_df']
        iv_df["feature"] = iv_df["feature"] + "_WOE"
        
        logger.info('-------------------DEBUG---------------')
        logger.info(iv_df.head())
        logger.info(iv_cat.head())
        logger.info(iv_num.head())
        

        logger.info(f"Final Train shape: {X_train.shape}")
        logger.info(f"Final Test shape : {X_test.shape}")

        logger.info("Top Numerical IV features:")
        logger.info(f"\n{iv_num.head()}")

        logger.info("Top Categorical IV features:")
        logger.info(f"\n{iv_cat.head()}")

  
        if X_train.isna().sum().sum() != 0:
            logger.error("NaNs detected in X_train after WOE transformation")
            raise ValueError("NaNs in X_train")

        if X_test.isna().sum().sum() != 0:
            logger.error("NaNs detected in X_test after WOE transformation")
            raise ValueError("NaNs in X_test")

        logger.info("Succesfully Binning is Done of features")
        logger.info('-------------------------------------------------------------------------------')
        logger.info("Feature Filteration has Started")
        
        selector = FeatureSelector(iv_threshold=0.02, corr_threshold=0.7, id_col=id_col)

        # Fit and get transformed data
        X_selected_train = selector.fit_transform(X_train,X_train_raw, runner, iv_num, iv_cat)
        X_selected_test = selector.transform(X_test)
        
        # save thiss
        bin_dir =  runner.fe_config.bin_dir
        os.makedirs(bin_dir,exist_ok=True)
        automatic_bin_dir = runner.fe_config.automatic_bin_dir
        os.makedirs(automatic_bin_dir,exist_ok=True)
        
        #paths
        iv_df_path = os.path.join(automatic_bin_dir,'iv_df.csv')
        cat_bin_df_path = os.path.join(automatic_bin_dir,'cat_bin_df.csv')
        
        numerical_feature_bins_dfs =  selector.feature_bins_num
        num_bin_path = os.path.join(automatic_bin_dir, "numerical_feature_bins.pkl")
        
        # save the artifacts
        with open(num_bin_path, "wb") as f:
            pickle.dump(numerical_feature_bins_dfs, f)
        logger.info(f'numerical_feature_bins_dfs Dumped succesfully Here :{num_bin_path}')
        
        ''' LOAD CODE :
        with open(num_bin_path, "rb") as f:
            feature_bins = pickle.load(f)
        '''
        
            
        # save the feature that are selected or filtered
        features = selector.selected_features_
        feature_list_path = os.path.join(
            runner.fe_config.automatic_bin_dir,
            "selected_features.json"
        )
        with open(feature_list_path, "w") as f:
            json.dump(features, f, indent=4)
        logger.info(f'selected_features  Dumped succesfully Here :{feature_list_path}')
        '''with open(feature_list_path, "r") as f:
            selected_features = json.load(f)
        '''

        iv_df.to_csv(iv_df_path,index=False)
        logger.info(f'iv_df Dumped succesfully Here:{iv_df_path}')
        
        cat_bin_df.to_csv(cat_bin_df_path,index=False)
        logger.info(f'cat_bin_df Dumped succesfully Here:{cat_bin_df_path}')

        
        logger.info('----SELECTED FEATURE AND IV----------------------------------------')
        logger.info(iv_df[iv_df['feature'].isin(features)])
    except Exception as e:
        raise MyException(e,sys,logger)
        