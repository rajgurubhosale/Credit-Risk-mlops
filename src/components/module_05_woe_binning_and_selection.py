from src.entity.data_transformation_artifact import *
from src.constants.data_transformation_constant import *
from src.entity.feature_engineering_artifact import *
from src.constants.feature_engineering_constant import *
from src.components.module_04_feature_quality_filter import orchastrator,load_clean_df
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
    '''Handle missing values for numerical variables'''
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
    Numerical WOE Binning using qcut pandas function
    - Fit on TRAIN only
    - Transform TRAIN / TEST safely
    """
    def __init__(self, id_col, special_codes=None, max_n_bins=20, min_prebin_size=0.05,prebinning_method='quantile'):
        '''
        Initialize WOE binning configuration.
        params
            id_col: id col from dataset
            special_codes: special values in dict to treat them as bin
            max_n_bins: Maximum number of final bins per variable
            min_prebin_size: Minimum proportion of observations per pre-bin

        '''
        self.special_codes = special_codes
        self.max_n_bins = max_n_bins
        self.min_prebin_size = min_prebin_size
        self.id_col = id_col
        self.bin_summary_ = {}
        self.bin_edges_ = {}
        #self.bin_assignment_ = {}
        self.iv_df_ = None


    def _manual_prebinning(self,
        df,
        feature,
        target,
        max_bins,
        min_bin_size,
        special_values=None,
        ):
        """
        Perform manual quantile-based pre-binning for a numerical
        feature for credit risk scorecards.
        
        - Params:    
        df : pd.DataFrame)
            The input dataset containing the feature to be binned.
        feature : str
            The name of the numerical feature to bin.
        target : pd.Series
            The binary target variable (0 = non-event, 1 = event) aligned with `df`.
        max_bins : int
            Maximum number of bins allowed for the feature.
        min_bin_size : float
            Minimum proportion of observations per bin (e.g., 0.05 = 5% of total data).
        special_values : list, optional
            List of values to treat as special codes (e.g., [-99999, -88888]).

       - Returns:
        final_bins : pd.Series
            Bin labels for each row; special values labeled as "SPECIAL_value".
        bin_table : pd.DataFrame
            Summary per bin with Count, Event, Non-event, Event rate, WoE, IV, and a totals row.
        bin_edges : list
            Numeric edges of the bins.
        total_iv : float
            Total Information Value of the feature.
        """

        if special_values is None:
            special_values = []
    
        y = target.loc[df.index]

        series = df[feature].copy()
        series = series.fillna(-99999)

        # Separate special & normal
        special_mask = series.isin(special_values)
        normal_mask = ~special_mask
        normal_series = series[normal_mask]

        final_bins = pd.Series(index=series.index, dtype="object")

        unique_vals = normal_series.nunique()

        # ---------------------------------
        # CASE 1: Low-cardinality numeric
        # ---------------------------------
        if unique_vals <= 6 and pd.api.types.is_numeric_dtype(normal_series):

            binned = normal_series.astype(str)
            bin_edges = sorted(normal_series.unique().tolist())

        # ---------------------------------
        # CASE 2: Continuous variable
        # ---------------------------------
        else:

            data_per_bin = normal_series.shape[0] * min_bin_size
            max_possible_bins = int(np.floor(normal_series.shape[0] / data_per_bin))
            n_bins = min(max_bins, max_possible_bins)

            try:
                binned, bin_edges = pd.qcut(
                    normal_series,
                    q=n_bins,
                    retbins=True,
                    duplicates="drop"
                )
            except ValueError:
                binned, bin_edges = pd.cut(
                    normal_series,
                    bins=min(5, normal_series.nunique()),
                    retbins=True
                )

        final_bins.loc[normal_mask] = binned.astype(str)

        # Handle special codes
        for val in special_values:
            final_bins.loc[series == val] = f"SPECIAL_{val}"

        # ---------------------------------
        # Create summary table
        # ---------------------------------
        bin_table = (
            pd.DataFrame({
                "feature": feature,
                "Bin": final_bins,
                "target": y
            })
            .groupby(["feature", "Bin"], dropna=False)
            .agg(
                Count=("target", "count"),
                Event=("target", "sum")
            )
            .reset_index()
        )

        bin_table["Non-event"] = bin_table["Count"] - bin_table["Event"]

        total_events = bin_table["Event"].sum()
        total_non_events = bin_table["Non-event"].sum()

        bin_table["Count (%)"] = bin_table["Count"] / len(df)
        bin_table["Event rate"] = bin_table["Event"] / bin_table["Count"]

        bin_table["dist_event"] = bin_table["Event"] / total_events
        bin_table["dist_non_event"] = bin_table["Non-event"] / total_non_events

        # WoE
        bin_table["WoE"] = np.where(
            (bin_table["Event"] == 0) | (bin_table["Non-event"] == 0),
            0,
            np.log(bin_table["dist_non_event"] / bin_table["dist_event"])
        )

        # IV
        bin_table["IV"] = (
            (bin_table["dist_non_event"] - bin_table["dist_event"])
            * bin_table["WoE"]
        )

        total_iv = bin_table["IV"].sum()

        bin_table = bin_table.drop(columns=["dist_event", "dist_non_event"])

        # ---------------------------------
        # Sort bins
        # ---------------------------------

        normal_bins = bin_table[~bin_table["Bin"].astype(str).str.startswith("SPECIAL")]
        special_bins = bin_table[bin_table["Bin"].astype(str).str.startswith("SPECIAL")]

        if normal_bins["Bin"].astype(str).str.contains(",").any():

            normal_bins = normal_bins.copy()
            normal_bins["lower"] = normal_bins["Bin"].apply(
                lambda x: float(str(x).split(",")[0].replace("(", "").replace("[",""))
            )

            normal_bins = normal_bins.sort_values("lower").drop(columns="lower")

        else:
            normal_bins = normal_bins.sort_values("Bin")

        bin_table = pd.concat([normal_bins, special_bins]).reset_index(drop=True)
        totals = pd.DataFrame({
        "feature": [feature],
        "Bin": [None],
        "Count": [bin_table["Count"].sum()],
        "Count (%)": [bin_table["Count (%)"].sum()],
        "Event": [bin_table["Event"].sum()],
        "Non-event": [bin_table["Non-event"].sum()],
        "Event rate": [
            bin_table["Event"].sum() / bin_table["Count"].sum()
            if bin_table["Count"].sum() != 0 else 0
        ],
        "WoE": [None],
        "IV": [bin_table["IV"].sum()],
        }, index=["Totals"])

        bin_table = pd.concat([bin_table, totals])
        
        #return final_bins, bin_table, list(bin_edges), total_iv
        del final_bins
        gc.collect()
        
        return bin_table, list(bin_edges), total_iv


    def fit(self, X, y):
        """ Fit WOE binning models using TRAINING data only
        fits the optbinning model per numerical variable  calculate the bin ,woe, and iv for features
        stores binning model.

        Params:
            X_train_num: pd.DataFrame containing only numerical features including id_col
            y_train : Binary target variable (event / non-event)

        """
        self.numerical_features = [c for c in X.columns if c != self.id_col]
        iv_list = []
        
        for feature in self.numerical_features:
            try:
                summary, edges, total_iv = self._manual_prebinning(
                    df=X,
                    feature=feature,
                    target=y,
                    max_bins=self.max_n_bins,
                    min_bin_size=self.min_prebin_size,
                    special_values=[-99999, -88888, -77777]
                )

                summary["Bin"] = summary["Bin"].astype(str)

                self.bin_summary_[feature] = summary
                self.bin_edges_[feature] = edges
                #self.bin_assignment_[feature] = bins
                iv_list.append({"feature": feature, "IV": total_iv})

            except Exception as e:
                logger.warning(f"Skipped {feature}: {e}")
                
                
        logger.info("Number of IV entries:", len(iv_list))
        
        self.iv_df = pd.DataFrame(iv_list).sort_values("IV", ascending=False).reset_index(drop=True)
        if self.iv_df.empty:
            logger.warning("No IV values computed — all features failed.")
            
        return self

    def transform(self, X):
        """
        Apply pre-fitted WOE binning rules to any dataset (train/test).
        Handles:
        - Special codes
        - Values outside training bins (clips to nearest bin)
        - Safe mapping for unseen bins

        Params:
            X: pd.DataFrame with numerical features (train/test)

        Returns:
            X_woe: WOE-transformed numerical features
        """

        woe_data = {}

        for feature in self.numerical_features:

            if feature not in self.bin_summary_:
                continue

            series = X[feature].copy()
            series = series.fillna(-99999)

            edges = self.bin_edges_[feature]
            summary = self.bin_summary_[feature].copy()

            summary = summary[summary["Bin"].notna()]
            summary["Bin"] = summary["Bin"].astype(str)

            woe_map = dict(zip(summary["Bin"], summary["WoE"]))

            is_interval = summary["Bin"].str.contains(",").any()

            if is_interval:
                series_clipped = series.clip(lower=edges[0], upper=edges[-1])

                binned = pd.cut(
                    series_clipped,
                    bins=edges,
                    include_lowest=True
                ).astype(str)

            else:
                # 🔥 FIX: enforce same dtype as training
                binned = series.astype(float).astype(int).astype(str)

            if self.special_codes is not None:
                for val in self.special_codes:
                    binned[series == val] = f"SPECIAL_{val}"

            woe_values = binned.map(woe_map)

            if woe_values.isna().any():
                woe_values = woe_values.fillna(summary["WoE"].mean())

            woe_data[feature] = woe_values

        X_woe = pd.DataFrame(woe_data, index=X.index).astype("float32")

        if self.id_col in X.columns:
            X_woe[self.id_col] = X[self.id_col]

        return X_woe

    def fit_transform(self, X_train_num, y_train):
        '''Fit WOE bins using training data and
        return WOE-transformed training features. call strictly on for the training data
        
        params:
            X_train_num: pd.DataFrame containing only numerical features including id_col
            y_train : Binary target variable (event / non-event)
            
        Returns:
            X_woe: WOE-transformed numerical features with consistent binning
        '''
        self.fit(X_train_num, y_train)
        return self.transform(X_train_num)
    
    
#-----------------------------------------------------------------
class CategoricalWOEBinner:
    """
    Categorical WOE Binning
    - One bin per category
    - Group rare categories into 'RARE'
    - Fit WOE values using training data only
    - Apply same mapping to validation / test data
    """
    def __init__(self, id_col, rare_threshold=0.01, ):
        '''
        Set basic parameters for categorical WOE binning.
        
        param:
            id_col: id_col: id col from dataset
            rare_threshold: Minimum frequency required to keep a category
        '''
        self.id_col = id_col
        self.rare_threshold = rare_threshold
        self.eps = 1e-6 # small value added to avoid division error

        self.categorical_features = []
        self.cat_woe_maps = {}
        self.iv_df = None
        
    def get_categorical_woe_bins(self,cat_woe_maps):
        ''' Create a readable summary of categorical WOE bins for the features.
            This function converts stored WOE mappings for categorical features
            into a tabular format.
            used for interpretation of the categorical features
            paramas:
                cat_woe_maps: Dictionary created during `fit()` that contains mapping of category -> WOE value :


            returns:
                records: ->pd.DataFrame
                  A DataFrame containing the complete WOE bin details for features
            '''
        records = []
        for feature, info in self.cat_woe_maps.items():
            stats = info["stats"]
            rare_cats = info["rare_categories"]

            for bin_name, s in stats.items():
                records.append({
                    "feature": feature,
                    "Bin": bin_name,
                    "Count": s["Count"],
                    "Event": s["Event"],
                    "Non-event": s["Non-event"],
                    "Count(%)": s["Count(%)"],
                    "Event rate": s["Event rate"],
                    "WoE": s["WoE"],
                    "IV": s["IV"],
                    "is_rare_bin": bin_name == "RARE",
                    "rare_categories": ", ".join(rare_cats) if bin_name == "RARE" else None
                })
            

        return pd.DataFrame(records)
    
    def fit(self, X_train_cat, y_train):
        """
        Fit WOE mappings using TRAINING data only.
            - Identify categorical features
            - group rare categories
            - calculate the woe and iv per category
            - store the mapping for later use

        Params:    
            X_train_cat: pd.DataFrame containing only categorical features including id_col
            y_train : Binary target variable (event / non-event)
            
        """
        self.categorical_features = [c for c in X_train_cat.columns if c != self.id_col]
        cat_iv_records = []

        for col in self.categorical_features:
            train_series = X_train_cat[col].fillna("MISSING")
            freq = train_series.value_counts(normalize=True)
            rare_categories = freq[freq < self.rare_threshold].index.tolist()

            # Group rare categories
            train_binned = train_series.where(~train_series.isin(rare_categories), "RARE")
            df_tmp = pd.DataFrame({"cat": train_binned, "target": y_train})

            # Aggregate counts
            agg = df_tmp.groupby("cat").agg(
                Count=("target", "count"),
                Event=("target", "sum")
            ).reset_index()
            agg["Non-event"] = agg["Count"] - agg["Event"]

            total_count = agg["Count"].sum()
            total_good = agg["Non-event"].sum()
            total_bad = agg["Event"].sum()

            # Distributions
            agg["Count(%)"] = agg["Count"] / total_count
            agg["Event rate"] = agg["Event"] / agg["Count"]
            agg["dist_good"] = agg["Non-event"] / total_good
            agg["dist_bad"] = agg["Event"] / total_bad

            # WOE
            agg["WoE"] = np.where(
                (agg["dist_good"] == 0) | (agg["dist_bad"] == 0),
                0.0,
                np.log(agg["dist_good"] / agg["dist_bad"])
            )

            # IV per bin
            agg["IV"] = (agg["dist_good"] - agg["dist_bad"]) * agg["WoE"]
            iv_value = agg["IV"].sum()

            # Stats for transform
            stats = {}
            for _, row in agg.iterrows():
                stats[row["cat"]] = {
                    "Count": row["Count"],
                    "Event": row["Event"],
                    "Non-event": row["Non-event"],
                    "Count(%)": row["Count(%)"],
                    "Event rate": row["Event rate"],
                    "WoE": row["WoE"],
                    "IV": row["IV"]
                }

            # Ensure RARE exists
            if "RARE" not in stats:
                stats["RARE"] = {
                    "Count": 0,
                    "Event": 0,
                    "Non-event": 0,
                    "Count(%)": 0.0,
                    "Event rate": 0.0,
                    "WoE": 0.0,
                    "IV": 0.0
                }

            woe_map = dict(zip(agg["cat"], agg["WoE"]))

            if "RARE" not in woe_map:
                woe_map["RARE"] = 0.0

            self.cat_woe_maps[col] = {
                "woe_map": woe_map,
                "rare_categories": rare_categories,
                "stats": stats
            }

            cat_iv_records.append({"feature": col, "IV": iv_value})

        self.iv_df = pd.DataFrame(cat_iv_records).sort_values(by="IV", ascending=False).reset_index(drop=True)
        return self

    def transform(self, X_cat):
        '''
        Apply learned WOE mappings to Data
            - uses the mapping learned in the fit method 
            - Handles unseen categories safely by mapping to 'RARE'
        Params:
        X_cat : pd.DataFrame
            Categorical features including id_col
        
        Returns:
            X_woe:->pd.DataFrame
            WOE-transformed categorical features
        '''
        
        X_woe = pd.DataFrame(index=X_cat.index)
        X_woe[self.id_col] = X_cat[self.id_col]

        for col in self.categorical_features:
            if col not in self.cat_woe_maps:
                continue

            woe_map = self.cat_woe_maps[col]["woe_map"]
            rare_categories = self.cat_woe_maps[col]["rare_categories"]

            test_series = X_cat[col].fillna("MISSING")
            test_binned = test_series.where(~test_series.isin(rare_categories), "RARE")
            test_binned = test_binned.where(test_binned.isin(woe_map.keys()), "RARE")
            X_woe[col] = test_binned.map(woe_map)

        return X_woe

    def fit_transform(self, X_train_cat, y_train):
        '''Fit WOE mappings and transform training data. use only on the training data'''
        self.fit(X_train_cat, y_train)
        return self.transform(X_train_cat)
#-----------------------------------------------------------------

class WOEFeatureSelector:
    '''
    Feature selection utility for credit risk modeling.
        1. Remove low-IV features
        2. Compute feature quality scores
        - Numerical: IV, missing rate, special value rate
        - Categorical: IV, missing rate, rare category rate
        3. Merge numerical + categorical scores
        4. Remove highly correlated features using feature quality score
    '''
    
    def __init__(self,iv_threshold=0.02,corr_threshold=0.7,id_col=None):
        '''
        Initialize feature selection thresholds.
        
        Params:    
            iv_threshold: Minimum IV required to keep a feature
            corr_threshold: Correlation threshold above which features are considered redundant
            id_col: id_col: id col from dataset

        '''
        self.iv_threshold = iv_threshold
        self.corr_threshold = corr_threshold
        self.id_col = id_col
        self.removed_low_iv_ = None
        self.removed_corr_ = None
    
    def iv_filteration(self, X_woe, iv_df):
        """
        Remove features with Information Value (IV) below threshold.
        Params:
            X_woe: pd.DataFrame
                DataFrame containing WOE-transformed features.
            iv_df: pd.DataFrame
                Dataframe cotanining the feature and its Information Value
        return:
            keep_features: List of WOE feature names that passed IV filtering
        """
        try:
            if self.id_col in X_woe.columns:
                X_woe = X_woe.drop(columns=[self.id_col])
                
            # keep features above threshold
            filt = iv_df["IV"] >= self.iv_threshold
            
            keep_features= iv_df.loc[filt,"feature"].tolist()

            self.removed_low_iv_ = iv_df.loc[
                iv_df["IV"] < self.iv_threshold, ["feature", "IV"]
            ]
                        
            logger.info(f"Kept Feature {len(keep_features)} features with IV > {self.iv_threshold}")           
            
            logger.info(f"Removed {self.removed_low_iv_.shape[0]} features with IV < {self.iv_threshold}")           
            
        except Exception as e:
            raise MyException(e,sys,logger)
        
        return keep_features


    def correlation_filter(self, X_train_woe, keep_features, iv_df):
        """
        Remove highly correlated features based on a threshold, 
        keeping the feature with the highest IV.
        Params:        
            X_train_woe : pd.DataFrame
                WOE-transformed features dataframe.
            keep_features : list
                List of features kept after IV filtering.
            iv_df : pd.DataFrame
                DataFrame containing IV values for features ('feature' and 'iv' columns).
        Returns:
            selected_features : list
                List of features selected after removing highly correlated features.
        """
        try:
            corr_threshold = self.corr_threshold
            X_filtered_woe = X_train_woe[keep_features].astype(np.float32)
            
            iv_df = iv_df.copy()
            iv_df.index = iv_df['feature']

            common_features = X_filtered_woe.columns.intersection(iv_df.index)
            X_filtered_woe = X_filtered_woe[common_features]
            iv_df = iv_df.loc[common_features]

            corr_matrix = X_filtered_woe.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            drop_features = set()

            for feature in upper.columns:
                if feature in drop_features:
                    continue

                correlated = upper.index[
                    (upper[feature] > corr_threshold) &
                    (~upper.index.isin(drop_features))
                ].tolist()

                if not correlated:
                    continue

                group = [feature] + correlated  # deterministic, no set()

                best_feature = iv_df.loc[group].sort_values("IV", ascending=False).index[0]

                for f in group:
                    if f != best_feature:
                        drop_features.add(f)

                logger.debug(f"Corr group: {group} | kept: {best_feature}")

            selected_features = [f for f in X_filtered_woe.columns if f not in drop_features]
            self.removed_corr_ = list(drop_features)

            logger.info(f"Total features before : {X_filtered_woe.shape[1]}")
            logger.info(f"Total features after  : {len(selected_features)}")
            logger.info(f"Dropped features      : {len(self.removed_corr_)}")
            logger.info(f"Removed {len(self.removed_corr_)} features due to high correlation (> {self.corr_threshold})")
            logger.info(f"Remaining feature list final shape is  {len(selected_features)}")

        except Exception as e:
            raise MyException(e, sys, logger)

        return selected_features

    
    def fit(self, X_train_woe,iv_df):
        """
        Fit the selector:
        - filter by IV
        - compute numerical and categorical feature quality scores
        - merge scores
        - filter correlated features
        """
        try:
            # IV filtering
            keep_features = self.iv_filteration(X_train_woe, iv_df)
            logger.info(f'IV filtering DONE ')
            
            # Correlation + IV  filtering
            self.selected_features_ = self.correlation_filter(X_train_woe,keep_features,iv_df)
            logger.info(f'Correlation filtering DONE succesfully')

        except Exception as e:
            raise MyException(e, sys, logger)
        
        return self 


    def transform(self, X):
        """
        Apply the fitted feature selection rules to a dataset learned from fit method.
        
        X : pd.DataFrame
            Input dataframe containing WOE-transformed features.
        Returns:
            pd.DataFrame: 
            Dataframe containing only the selected features that are available
            in the input dataset.
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

  
    def fit_transform(self,X_train_woe,iv_df):
        ''' 
        Fit the FeatureSelector on training data and return the transformed dataset.
        
        Return:
            pd.DataFrame:
            Filtered WOE-transformed training dataset containing only
            the selected features.
        '''
        self.fit(X_train_woe, iv_df)
        return self.transform(X_train_woe)
    
class WOETransformer:
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
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Fitting WOE bins (TRAIN only)")

        X = handle_missing_values(X, self.id_col)

        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        self.num_cols_ = list(dict.fromkeys([self.id_col] + self.num_cols_))
        self.cat_cols_ = list(dict.fromkeys([self.id_col] + self.cat_cols_))
        
        logger.info("Fitting & transforming Numerical WOE (TRAIN)")
        self.num_binner.fit(X[self.num_cols_], y)
        logger.info("Fitting & transforming Categorical WOE (TRAIN)")
        self.cat_binner.fit(X[self.cat_cols_], y)

        self.iv_num_ = self.num_binner.iv_df.copy()
        self.iv_cat_ = self.cat_binner.iv_df.copy()
        self.iv_df_ = (pd.concat([self.iv_num_, self.iv_cat_], axis=0)
                    .reset_index(drop=True))
        logger.info("WOE IS Fit on the train data")
        
        return self
    
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying WOE transformation")

        X = handle_missing_values(X, self.id_col)

        X_num_woe = self.num_binner.transform(X[self.num_cols_])
        X_cat_woe = self.cat_binner.transform(X[self.cat_cols_])
        
        X_final = pd.concat(
            [
                X_num_woe.drop(columns=[self.id_col]),
                X_cat_woe.drop(columns=[self.id_col])
            ],
            axis=1
        ).astype(np.float32)
        
        del X_num_woe, X_cat_woe
        gc.collect()
        
        logger.info(f"Transformation completed | Shape: {X_final.shape}")
        return X_final

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
    
class WOEArtifactManager:
    
    def __init__(self):
        self.fe_config = FeatureEngConfig()

        self.bin_dir = self.fe_config.bin_dir
        self.prebin_dir = self.fe_config.prebin_dir
        self.splits_dir = self.fe_config.splits_dir
        self._create_dirs()
    
    def _create_dirs(self):
        os.makedirs(self.bin_dir, exist_ok=True)
        os.makedirs(self.prebin_dir, exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)

   

    def save_iv_df(self, iv_df):
        path = os.path.join(self.prebin_dir, "iv_df.csv")
        if os.path.exists(path):
            logger.warning(f"Overwriting existing IV file at {path}")
        iv_df.to_csv(path, index=False)
        logger.info(f"IV dataframe saved at: {path}")
    
    def save_splits(self, X_train, X_test, y_train, y_test):
        X_train.to_csv(os.path.join(self.splits_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.splits_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.splits_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.splits_dir, "y_test.csv"), index=False)
        logger.info(f" splitted data saved at: {self.splits_dir}")
        

        
    def save_categorical_bins(self, cat_bin_df):
        path = os.path.join(self.prebin_dir, "cat_bin_df.csv")
        cat_bin_df.to_csv(path, index=False)
        logger.info(f"Categorical WOE bins saved at: {path}")
    
    def save_final_data(self,X_train,X_test):
        
        X_train_path = os.path.join(self.prebin_dir, "X_train_selected_woe.csv")
        X_test_path = os.path.join(self.prebin_dir, "X_test_selected_woe.csv")
        X_train.to_csv(X_train_path,index =False)
        X_test.to_csv(X_test_path,index =False)
        logger.info(f'tHe Final Train_woe and Test_woe saved after feature selection:{ self.prebin_dir}  ')
        

    def save_numerical_bins(self, numerical_feature_bins):
        path = os.path.join(
            self.prebin_dir,
            "numerical_feature_bins.pkl"
        )
        with open(path, "wb") as f:
            pickle.dump(numerical_feature_bins, f)
        
        
        logger.info(f"Numerical feature bins dumped at: {path}")
        
        
    def save_selected_features(self, selected_features):
        path = os.path.join(
            self.prebin_dir,
            "selected_features.json"
        )
        with open(path, "w") as f:
            json.dump(selected_features, f, indent=4)

        logger.info(f"Selected features saved at: {path}")
        
class WOEBinningPipeline:
    def __init__(self, id_col, target_col):
        self.id_col = id_col
        self.target_col = target_col
        
        self.fe_config = FeatureEngConfig()
        self.artifact_manager = WOEArtifactManager()
        # Objects that will be set later
        self.woe_runner = None
        self.fe_selector = None
        
    def _split_data(self, df):

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            stratify=y,
            test_size=self.fe_config.test_size,
            random_state=self.fe_config.default_random_state
        )

        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Test shape : {X_test.shape}")
        
        logger.info('Data Splitting Done')
        
        del X,y
        gc.collect()
        
        return X_train, X_test, y_train, y_test
    
    def _feature_quality_filter(self, X_train, X_test):

        feature_quality_csv_path = orchastrator(X_train)
        feature_quality_df = pd.read_csv(feature_quality_csv_path)

        X_train_clean = load_clean_df(X_train, feature_quality_df)
        X_test_clean = load_clean_df(X_test, feature_quality_df)

        logger.info(f"After feature quality filtering: {X_train_clean.shape}")

        return X_train_clean, X_test_clean
    
    def _apply_woe_transformation(self, X_train, X_test, y_train):

        self.woe_runner = WOETransformer(self.id_col, self.target_col)
        
        
        X_train_woe = self.woe_runner.fit_transform(X_train, y_train)
        del X_train
        gc.collect()
        
        X_test_woe = self.woe_runner.transform(X_test)
        del X_test
        gc.collect()
        
        X_train_woe = X_train_woe.astype("float32")
        X_test_woe  = X_test_woe.astype("float32")
        
        self.iv_df = self.woe_runner.iv_df_

        if X_train_woe.isna().sum().sum() != 0:
            raise ValueError("NaNs detected after WOE transformation")
        
        logger.info(f'WOE Transformation Applied')
        logger.info(f"WOE Train shape: {X_train_woe.shape}")
        logger.info(f"WOE Test shape : {X_test_woe.shape}")

        return X_train_woe, X_test_woe
    
    def _feature_selection(self, X_train_woe, X_test_woe):

        self.fe_selector = WOEFeatureSelector(id_col=self.id_col)

        X_selected_train = self.fe_selector.fit_transform(
            X_train_woe,
            self.iv_df
        )

        X_selected_test = self.fe_selector.transform(X_test_woe)
        logger.info(f'Feature Selection Applied')
        

        logger.info(f"Selected features count: {len(self.fe_selector.selected_features_)}")

        return X_selected_train, X_selected_test
    
    def _save_artifacts(self):

        self.artifact_manager.save_selected_features(self.fe_selector.selected_features_)
        self.artifact_manager.save_iv_df( self.iv_df)

        logger.info("Artifacts saved successfully")
        
    def run(self):
        try:
            logger.info("Pipeline execution started")
    
            data_path = r"D:\home loan credit risk\artifact\interim\main_df_transformed.csv"
            logger.info(f"Loading dataset from: {data_path}")
            
            df = pd.read_csv(data_path)
    
            if "YEARS_EMPLOYED" in df.columns:
                df['YEARS_EMPLOYED'] = df['YEARS_EMPLOYED'].replace({-1000.67:-99999})
            
            for col in df.select_dtypes("int64"):
                df[col] = pd.to_numeric(df[col], downcast="integer")
                
            float_cols = df.select_dtypes(include=["float64"]).columns
            df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")

            logger.info(f'Main df shape:{df.shape} ')
            
            X_train,X_test,y_train,y_test = self._split_data(df)
            
            del df
            gc.collect()
            
            #save the splits of the data
            self.artifact_manager.save_splits(X_train,X_test,y_train,y_test)
            
            X_train,X_test = self._feature_quality_filter(X_train,X_test)
            
            X_train_woe,X_test_woe =  self._apply_woe_transformation(X_train,X_test,y_train)
            
            # run this method to genrate the numerical feature bins dataframe rules
            numerical_feature_bins = self.woe_runner.num_binner.bin_summary_
            
            self.artifact_manager.save_numerical_bins(numerical_feature_bins)
            
            
            # save the categorical bins
            cat_bin_df = self.woe_runner.cat_binner.get_categorical_woe_bins(
            self.woe_runner.cat_binner.cat_woe_maps)
            self.artifact_manager.save_categorical_bins(cat_bin_df)

            X_selected_train, X_selected_test = self._feature_selection(X_train_woe,X_test_woe)
            self.artifact_manager.save_final_data(X_selected_train,X_selected_test)
            
            logger.info("IV features:")
            logger.info(self.iv_df["feature"].head())
            
            logger.info(self.iv_df["feature"].head())
            
            logger.info("Selected features:")
            logger.info(f"selected features and iv: {self.iv_df[self.iv_df['feature'].isin(self.fe_selector.selected_features_)]}")
            self._save_artifacts()
            
            logger.info("Pipeline execution completed successfully")

        except Exception as e:
            raise MyException(e,sys,logger)
        
        
if __name__ == '__main__':
    fe_pipeline = WOEBinningPipeline(id_col,target_col)
    fe_pipeline.run()
    