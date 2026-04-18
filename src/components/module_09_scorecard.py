from src.entity.artifact_entity import FeatureBinMergingArtifact,ModelTrainigArtifact,FeatureEngArtifact,ScorecardArtifact
import numpy as np
import joblib
import pickle
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import gc
import pandas as pd
import json
from src.exception import MyException
from src.logger import config_logger
import sys
logger = config_logger('module_09_scorecard')


class Scorecard:
    '''
    Builds a PDO-scaled credit scorecard from trained logistic regression model
    and WoE-based bins
    Convert the model log odds output to scores
    output:
    - final_scorecard_rules:
    - final_deciles_scorecard:
    - scorecard_scaling_params:
    
    '''
    def __init__(self):
        self.merge_artifact   = FeatureBinMergingArtifact()
        self.model_artifact   = ModelTrainigArtifact()
        self.fe_artifact    = FeatureEngArtifact()
        self.scorecard_artifact =ScorecardArtifact()

        # ── scorecard scaling constants ──────────────────────────────
        self.BASE_SCORE = 600
        self.BASE_ODDS  = 50
        self.PDO        = 20
        
        self.model = None
        self.final_features = None
        self.intercept_points = None
        self.scorecard_df = None
        self.feature_type_col = 'feature_type' 

        self.numerical_lookup = {}
        self.categorical_lookup = {}
        
    def _compute_scaling_params(self):
        '''
        Compute scorecard scaling constants using PDO method.
        Converts logistic regression intercept into scorecard
        intercept points using PDO scaling logic..
        
        return:
        - FACTOR
        - INTERCEPT_POINTS  
        
        '''
        self.model_intercept  = float(self.model.intercept_[0])

        FACTOR = self.PDO / np.log(2)
        OFFSET = self.BASE_SCORE + FACTOR * np.log(self.BASE_ODDS)
        INTERCEPT_POINTS = OFFSET - FACTOR * self.model_intercept
        self.intercept_points = INTERCEPT_POINTS

        return FACTOR, INTERCEPT_POINTS
    

    def numerical_score_rules(self,num_bins, iv_df, feature_importance, FACTOR):
        '''
        Build scorecard rules for numerical features.
        For each selected numerical feature:
        - Attaches IV from iv_df (feature iv)
        - Computes per-bin score points:  score_points = -(WoE * coefficient) * FACTOR
        - Drops the 'Totals' summary row added by the binning module
        
        Returns:
        num_score_rules: Scorecard rules for all numerical features.
            [feature, Bin, WoE, Event rate, Count (%), Feature IV,
            feature_coeff_model, score_points]'''

        final_summary = {}

        for feature, summary_data in num_bins.items():
            if feature not in self.final_features:
                continue

            feature_iv   = iv_df[iv_df['feature'] == feature]['IV'].values[0]
            feature_coef = feature_importance[feature_importance['feature'] == feature]['coeff'].values[0]

            summary_data['Feature IV'] = feature_iv

            df = summary_data[['feature', 'Bin', 'WoE', 'Event rate', 'Count (%)', 'Feature IV']].copy()

            df['feature_coeff_model'] = feature_coef
            df['score_points']        = 0.0

            n_bin = df['Bin'].iloc[:-1].shape[0]

            for bin_no in range(n_bin):
                idx       = df.index[bin_no]
                bin_score = (df.loc[idx, 'WoE'] * feature_coef) * (-FACTOR) 
                df.loc[idx, 'score_points'] = bin_score

            final_summary[feature] = df

        num_score_rules = pd.concat(final_summary.values(), ignore_index=True)

        return num_score_rules 
        
        
    def categorical_score_rules(self,cat_bin_df, iv_df, feature_importance, FACTOR):

        ''' 
        Build scorecard rules for categorical features.

        For each selected categorical feature:
        - Filters bins to only final selected features
        - Add IV and model coefficient
        - Computes per-bin score points: -(WoE * coeff) * FACTOR
            
        return:
        - Cat_scorecard: Scorecard rules for categorical features.
            [feature, Bin, WoE, Event rate, Count (%),
            rare_categories, Feature IV, feature_coeff_model, score_points]
        '''
    
        cat_bin_df   = cat_bin_df.rename(columns={'Count(%)': 'Count (%)'})
        cat_features = cat_bin_df['feature'].unique().tolist()
        results      = []

        for feature in self.final_features:
            if feature not in cat_features:
                continue

            feature_coef = feature_importance[feature_importance['feature'] == feature]['coeff'].values[0]
            feature_iv   = iv_df[iv_df['feature'] == feature]['IV'].values[0]

            temp = cat_bin_df[cat_bin_df['feature'] == feature][['feature', 'Bin', 'WoE', 'Event rate', 'Count (%)', 'rare_categories']].copy()

            temp['Feature IV']          = feature_iv
            temp['feature_coeff_model'] = feature_coef
            temp['score_points']        = (temp['WoE'] * feature_coef) * (-FACTOR)

            results.append(temp)

        return pd.concat(results, ignore_index=True) 



    def build_scorecard_rules(self,cat_bin_df,num_bins,iv_df, feature_importance):
        '''
        Combine numerical and categorical scorecard rules.
        build numerica_rules,categorical_rules
        add fetaure_types for numerical,categorical rules df
        then combine and create one final scorecard rules df
        
        returns:
        - final_scorecard_rules: scores assigend for each feature and its bins df
        - FACTOR
        - INTERCEPT_POINTS
        '''

        FACTOR,  INTERCEPT_POINTS = self._compute_scaling_params()

        self.num_scorecard_rules = self.numerical_score_rules(num_bins,iv_df, feature_importance, FACTOR)
        self.num_scorecard_rules = self.num_scorecard_rules[self.num_scorecard_rules['Bin'] != 'None']
        self.cat_scorecard_rules = self.categorical_score_rules(cat_bin_df,iv_df, feature_importance, FACTOR)
        self.cat_scorecard_rules = self.cat_scorecard_rules[self.cat_scorecard_rules['Bin'] != 'None']

        
        #   since this column in cat scorecard therefore create here to avoid merging error        
        self.num_scorecard_rules['rare_categories'] = None

        self.num_scorecard_rules['feature_type'] = 'numerical'
        self.cat_scorecard_rules['feature_type'] = 'categorical'

        # one scorecard to rule them all
        final_scorecard_rules = pd.concat(
            [self.num_scorecard_rules, self.cat_scorecard_rules], axis=0
        ).reset_index(drop=True)
        

        return final_scorecard_rules, FACTOR, INTERCEPT_POINTS
    

    def build_numerical_lookup(self):
        num_lookup = {}
        for feature, grp in self.num_scorecard_rules.groupby('feature'):
            temp = grp.copy()
            
            interval_mask = temp['Bin'].str.startswith('(') | temp['Bin'].str.startswith('[')
            special_mask  = temp['Bin'].str.startswith('SPECIAL_')
            discrete_mask = ~interval_mask & ~special_mask

            interval_df = temp[interval_mask].copy()
            special_df  = temp[special_mask].copy()
            discrete_df = temp[discrete_mask].copy()

            # --- Interval bins: IntervalIndex → score_points Series ---
            interval_series = pd.Series(dtype=float)
            if not interval_df.empty:
                interval_df['interval_bin'] = (interval_df['Bin']
                                                .str[1:-1]
                                                .str.split(',')
                                                .apply(lambda x: tuple(map(float, x))))
                interval_index  = pd.IntervalIndex.from_tuples(interval_df['interval_bin'], closed='right')
                interval_series = pd.Series(interval_df['score_points'].values, index=interval_index)

            # --- Discrete bins: numeric value → score_points ---
            discrete_series = pd.Series(dtype=float)
            if not discrete_df.empty:
                numeric_mask    = pd.to_numeric(discrete_df['Bin'], errors='coerce').notna()
                discrete_df     = discrete_df[numeric_mask]
                discrete_series = pd.Series(discrete_df['score_points'].values,
                                            index=pd.to_numeric(discrete_df['Bin']))

            # --- Special bins: extract the number → score_points ---
            # e.g. SPECIAL_-99999 → {-99999: score}
            special_dict = {}
            if not special_df.empty:
                for _, row in special_df.iterrows():
                    key = float(row['Bin'].replace('SPECIAL_', ''))  # -99999, -77777 etc.
                    special_dict[key] = row['score_points']

            num_lookup[feature] = {
                'interval': interval_series,   # pd.Series with IntervalIndex
                'discrete': discrete_series,   # pd.Series with numeric index
                'special':  special_dict       # {-99999: score, -77777: score, ...}
            }
        return num_lookup


    def build_categorical_lookup(self):
        cat_lookup = {}
        for feature, grp in self.cat_scorecard_rules.groupby('feature'):
            cat_lookup[feature] = dict(zip(grp['Bin'], grp['score_points']))
            
        return cat_lookup
    
    def _single_get_numerical_column_score(self,feature, value): 
        
        feature_lookup = self.numerical_lookup[feature]
        
        # Check special values first (-99999, -77777 etc.)
        if value in feature_lookup['special']:
            return feature_lookup['special'][value]
        
        # Check interval bins
        if not feature_lookup['interval'].empty:
            
            try:
                feature_low = feature_lookup['interval'].index[0].left
                feature_high = feature_lookup['interval'].index[-1].right
                
                if value < feature_low:
                    return feature_lookup['interval'].iloc[0]
                elif value > feature_high:                
                    return feature_lookup['interval'].iloc[-1]
                else:
                    return feature_lookup['interval'][value]
            except KeyError:
                
                pass
        
        # Check discrete bins (1, 2, 3 etc.)
        if not feature_lookup['discrete'].empty:
            try:
                if value < feature_lookup['discrete'].index.min():
                    return feature_lookup['discrete'].iloc[0]
                elif value > feature_lookup['discrete'].index.max():
                    return feature_lookup['discrete'].iloc[-1]
                else:
                    return feature_lookup['discrete'][value]
            except KeyError:
                pass
        
        return np.nan

  
    def _single_get_cat_score(self,feature, value):
        feature_lookup = self.categorical_lookup[feature]
        
        # direct lookup, fallback to RARE if unseen category
        return feature_lookup.get(value, feature_lookup.get('RARE', np.nan))
    
    def _single_score_applicant(self,user_info: dict):


        total_score = 0
        breakdown   = {}
        
        # 1. Numerical features
        for feature in self.numerical_lookup.keys():
            if feature in user_info:
                value = user_info[feature]
                
                # if value is NaN → treat as special -99999
                if pd.isna(value):
                    value = -99999.0
                
                score = self._single_get_numerical_column_score(feature, value)
                
                if not pd.isna(score):       
                    total_score        += score
                    breakdown[feature]  = score
        
        # 2. Categorical features
        for feature in self.categorical_lookup.keys():
            if feature in user_info:
                value = user_info[feature]
                
                # if value is NaN → treat as MISSING
                if pd.isna(value):
                    value = 'MISSING'
                
                score = self._single_get_cat_score(feature, value)
                
                if not pd.isna(score):        
                    total_score        += score
                    breakdown[feature]  = score

        total_score = self.intercept_points + total_score
                
        return {
            'total_score': round(total_score, 4),
            'breakdown':   breakdown
        }
    
    def _batch_get_num_score(self,feature, col):
    
        # clean the column
        col = pd.Series(col).fillna(-99999).astype(float)
        
        lookup  = self.numerical_lookup[feature]
        result  = pd.Series(np.nan, index=col.index)

        # step 1 — special values (e.g. -99999 for missing)
        for val, pts in lookup['special'].items():
            result[col == val] = pts

        # step 3 — clamp anything out of range to the nearest edge bin
        if not lookup['interval'].empty:
            
            for interval, pts in lookup['interval'].items():
                mask = result.isna() & (col > interval.left) & (col <= interval.right)
                result[mask] = pts
                
            low      = lookup['interval'].index[0].left
            high      = lookup['interval'].index[-1].right
            low_pts  = lookup['interval'].iloc[0]
            high_pts  = lookup['interval'].iloc[-1]
            result[result.isna() & (col <= low)] = low_pts
            result[result.isna() & (col >= high)] = high_pts
        
        if not lookup['discrete'].empty:
            
            discrete_map = lookup['discrete']

            # direct mapping
            mapped = col.map(discrete_map)

            result[result.isna()] = mapped[result.isna()]

            # clamp outside discrete range
            low_val  = discrete_map.index.min()
            high_val = discrete_map.index.max()

            low_pts  = discrete_map.iloc[0]
            high_pts = discrete_map.iloc[-1]

            result[result.isna() & (col < low_val)] = low_pts
            result[result.isna() & (col > high_val)] = high_pts

        
        # step 4 — anything still missing gets 0
        return result.fillna(0.0)
    
    def _batch_get_cat_score(self,feature, col):
        
        # clean the column — convert everything to plain string
        col = pd.Series(col).astype(str).str.strip()
        col = col.fillna('MISSING')
        col = col.replace({'nan': 'MISSING', '': 'MISSING', 'NaN': 'MISSING'})

        lookup   = self.categorical_lookup[feature]
        rare_pts = lookup.get('RARE', 0.0)

        # if category is known return its points, otherwise return RARE points
        return col.map(lambda x: lookup.get(x, rare_pts))

    def _batch_genrate_scores(self,X):
        
        logger.info(f'Score Genration Started')
        
        score_matrix = pd.DataFrame(index=X.index)

        for feature in self.numerical_lookup:
            if feature not in X.columns:
                print(f"[SKIP] {feature}")
                continue
            score_matrix[feature] = self._batch_get_num_score(feature, X[feature])

        for feature in self.categorical_lookup:
            if feature not in X.columns:
                print(f"[SKIP] {feature}")
                continue
            score_matrix[feature] = self._batch_get_cat_score(feature, X[feature])

        X['credit_score'] = (self.intercept_points + score_matrix.sum(axis=1)).round(2)

        return X
    
    def build_scorecard(self,scored_df):
        
        good = scored_df[scored_df['TARGET'] == 0]['credit_score']
        bad  = scored_df[scored_df['TARGET'] == 1]['credit_score']
        
        logger.info(f"Good borrowers — mean: {good.mean()}, std: {good.std()}")
        logger.info(f"Bad borrowers  — mean: {bad.mean()},  std: {bad.std()}")
        logger.info(f"Separation     — diff: {good.mean() - bad.mean()} points")
        

        scored_df['score_decile'] = pd.qcut(scored_df['credit_score'], q=10, duplicates='drop')
        decile_summary = scored_df.groupby('score_decile')['TARGET'].agg(['mean','count'])
        decile_summary.columns = ['default_rate','count']
        decile_summary['default_rate_pct'] = (decile_summary['default_rate'] * 100).round(2)
        
        
        decile_summary['bad_count']  = (decile_summary['default_rate'] * decile_summary['count']).round(0)
        decile_summary['good_count'] = decile_summary['count'] - decile_summary['bad_count']
        
        decile_summary['bad_capture_rate_cumsum'] = (
            decile_summary['bad_count'].cumsum() / decile_summary['bad_count'].sum() * 100
            ).round(2)
        decile_summary['lift'] = (
            decile_summary['default_rate']
            / scored_df['TARGET'].mean()
        )
        
        decile_summary = decile_summary.reset_index()
        ks   = ks_2samp(good, bad).statistic
        
        metrics = {
            "ks":   round(ks, 4),
        }
        logger.info(f"KS: {metrics['ks']}")

        return decile_summary, metrics
    
    
    def orchestrate(self):
        ''''
        Main pipeline controller for scorecard generation.
           1. Loads trained model
            2. Builds scorecard rules
            3. Saves scorecard outputs
            4. Scores training dataset
            5. Generates decile performance table
            6. Saves all final outputs
        Returns:
        
        - final_scorecard_rules : DataFrame
            Final scorecard rule table.

        - X_train_scored : DataFrame
            Training data with credit scores.

        - decile_table : DataFrame
            Scorecard performance summary.

        '''
        try:
            
            logger.info("Loading trained model bundle")

            bundle              = joblib.load(self.model_artifact.model_path)
            self.model          = bundle['model']
            self.final_features = bundle['features']
            logger.info(f"Model loaded successfully with "f"{len(self.final_features)} features")
            
            
            logger.info("Loading IV , feature importance , num bins, cat_bin_df files")
 
            iv_df              = pd.read_csv(self.fe_artifact.iv_df_path)
            feature_importance = pd.read_csv(self.model_artifact.feature_importance_path)

            num_bins_path = self.merge_artifact.final_selected_num_bins_path

            with open(num_bins_path,'rb') as f:
                num_bins = pickle.load(f)
                
            cat_bin_df = pd.read_csv(self.fe_artifact.cat_bin_df_path)
            
            
            logger.info("Loading IV,feature_importance,num_bins ,cat_bin_df completed")
            
                
            logger.info("Building scorecard rules")

            final_scorecard_rules, FACTOR, INTERCEPT_POINTS = self.build_scorecard_rules(cat_bin_df,num_bins,iv_df, feature_importance)
            
            final_scorecard_rules.to_csv(self.scorecard_artifact.final_scorecard_rules_path,index=False)
            logger.info(f'Final scorecard rules are saved here :{self.scorecard_artifact.scorecard_dir}')
            
            
            self.num_scorecard_rules.to_csv(self.scorecard_artifact.scorecard_numerical_rules,index=False)
            self.cat_scorecard_rules.to_csv(self.scorecard_artifact.scorecard_categorical_rules,index=False)
            
            logger.info(f'Numerical and Categorical rules are saved here :{self.scorecard_artifact.scorecard_dir}')
            
            
            
            self.numerical_lookup = self.build_numerical_lookup()
            
            
            joblib.dump(
                self.numerical_lookup,
                self.scorecard_artifact.scorecard_numerical_lookup
            )
            logger.info(f'Numerical Lookup dumped at :{self.scorecard_artifact.scorecard_dir}')
            
            
            self.categorical_lookup = self.build_categorical_lookup()
            
            joblib.dump(
                self.categorical_lookup,
                self.scorecard_artifact.scorecard_categorical_lookup
            )
            
            logger.info(f'Categorical Lookup dumped at :{self.scorecard_artifact.scorecard_dir}')
                   
            X_train = pd.read_csv(self.fe_artifact.data_splits_x_train_path,usecols=self.final_features)
            y_train = pd.read_csv(self.fe_artifact.data_splits_y_train_path).squeeze()
            logger.info(f'X_train and y_train is loaded')
            
            train_scored = self._batch_genrate_scores(X_train)
            train_scored['TARGET'] = y_train

            logger.info(f'Scores for train data are genrated')

            train_scored.to_csv(self.scorecard_artifact.train_score_df_path,index=False)
            
            logger.info(f'Saved the train+scores in {self.scorecard_artifact.scorecard_dir}')
            
            train_scored = train_scored[['credit_score','TARGET']].copy()
            
            final_scorecard,metrics = self.build_scorecard(train_scored)
            logger.info(f'Final ScoreCard is BUILT')
            logger.info(final_scorecard)
            
            
            final_scorecard.to_csv(self.scorecard_artifact.final_scorecard_table_path,index=False)
            logger.info(f'final scorecard deciles saved to :{self.scorecard_artifact.scorecard_dir}')
            
            with open(self.scorecard_artifact.scorecard_metrics_path,'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved at: {self.scorecard_artifact.scorecard_metrics_path}")

            
            # save scaling params
            params = {
                "FACTOR"           : float(FACTOR),
                "INTERCEPT_POINTS" : float(INTERCEPT_POINTS),
                "BASE_SCORE"       : self.BASE_SCORE,
                "BASE_ODDS"        : self.BASE_ODDS,
                "PDO"              : self.PDO,
                "intercept"        :self.intercept_points
            }
            
            with open(self.scorecard_artifact.scorecard_scaling_params_path, 'w') as f:
                json.dump(params, f)
            logger.info(f"Scaling parameters saved at: {self.scorecard_artifact.scorecard_scaling_params_path}")
            
            
      
        except Exception as e:
            raise MyException(e,sys,logger)
    
if __name__ =='__main__':
    logger.info("Starting Scorecard Orchestration Pipeline")
    
    scorecard = Scorecard()
    scorecard.orchestrate()
    logger.info("Scorecard Orchestration Completed Successfully")