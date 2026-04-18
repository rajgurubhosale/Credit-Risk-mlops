from sklearn.linear_model import LogisticRegression
from src.entity.artifact_entity import FeatureBinMergingArtifact, FeatureEngArtifact,ModelTrainigArtifact
from src.entity.config_entity import FeatureEngConfig
import pandas as pd
import numpy as np
from src.utils.main_utils import read_yaml_file
import joblib
from src.logger import config_logger
from src.exception import MyException
import sys
import json

logger = config_logger('module_07_model_training.py')

class ModelTraining:    
    
    def __init__(self):
        
        self.bin_merge_artifact = FeatureBinMergingArtifact()
        self.fe_artifact = FeatureEngArtifact()
        self.fe_config = FeatureEngConfig()
        self.model_artifact = ModelTrainigArtifact()
        params = read_yaml_file(self.fe_config.params_path,logger)
        self.params = params.get('model_training')
        
    def train_model(self,X_train,y_train):
        
        model = LogisticRegression(
                penalty=self.params['penalty'],         
                solver='liblinear',   
                C=self.params['c'],               
                class_weight=self.params['class_weight'],
                max_iter=self.params['max_iter'],
                random_state=self.params['random_state']
            )
        
        model.fit(X_train,y_train)
        
        return model

    def orchestrate(self):

        try:
            logger.info("Starting Model Training Pipeline")

            # Load final selected features
            logger.info("Loading final selected features")

            final_selected_features_path = (self.bin_merge_artifact.final_model_features)
            with open(final_selected_features_path,'rb') as file:
                final_selected_features = json.load(file)
            

            logger.info(
                f"Number of selected features: {len(final_selected_features)}"
            )

            # Load X_train
            logger.info("Loading X_train dataset")

            X_train = pd.read_csv(
                self.bin_merge_artifact.X_train_final_path,
                usecols=final_selected_features
            )

            logger.info(
                f"X_train loaded with shape: {X_train.shape}"
            )

            # Load y_train
            logger.info("Loading y_train dataset")

            y_train = pd.read_csv(
                self.fe_artifact.data_splits_y_train_path
            )

            y_train = y_train.values.ravel()

            logger.info(
                f"y_train loaded with shape: {y_train.shape}"
            )

            # Train model
            model = self.train_model(X_train, y_train)

            # Save feature importance
            logger.info("Saving feature importance")

            feature_importance = pd.DataFrame({
                'feature': model.feature_names_in_,
                'coeff': model.coef_[0]
            })

            feature_importance = feature_importance.reset_index(drop=True)

            feature_importance.to_csv(
                self.model_artifact.feature_importance_path,
                index=False
            )

            logger.info(
                f"Feature importance saved at: "
                f"{self.model_artifact.feature_importance_path}"
            )

            # Save model
            logger.info("Saving trained model")

            model_path = self.model_artifact.model_path

            joblib.dump(
                {
                    "model": model,
                    "features": final_selected_features
                },
                model_path
            )

            logger.info(
                f"Model saved successfully at: {model_path}"
            )

            logger.info("Model Training Pipeline completed successfully")

        except Exception as e:
            raise MyException(e,sys,logger)
if __name__ == '__main__':
    logger.info("Starting Model Training Script")

    model_train_obj = ModelTraining()
    model_train_obj.orchestrate()

    logger.info("Script execution completed")