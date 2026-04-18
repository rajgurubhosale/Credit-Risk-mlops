from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from src.exception import MyException
from src.logger import config_logger
import sys

logger = config_logger('module_08_model_evalution')

from src.entity.artifact_entity import (
    FeatureBinMergingArtifact,
    FeatureEngArtifact,
    ModelTrainigArtifact,
    ModelEvalArtifact
)

class ModelEvaluation:

    def __init__(self):

        self.bin_merge_artifact = FeatureBinMergingArtifact()
        self.fe_artifact = FeatureEngArtifact()
        self.model_artifact = ModelTrainigArtifact()
        self.eval_artifact = ModelEvalArtifact()


    def calculate_ks(self, y_true, y_prob):
        ''' calculate the ks value'''

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        ks = np.max(tpr - fpr)

        return ks

    def calculate_gini(self, auc):
        ''' calculate the gini value'''

        return 2 * auc - 1

    def save_roc_curve(self,y_train, train_prob,y_test, test_prob,roc_path):
        # Train ROC
        fpr_train, tpr_train, _ = roc_curve(y_train,train_prob)

        # Test ROC
        fpr_test, tpr_test, _ = roc_curve( y_test,test_prob)

        plt.figure()

        plt.plot(fpr_train,tpr_train,label="Train ROC")

        plt.plot(fpr_test,tpr_test,label="Test ROC")
        plt.plot(fpr_train,tpr_train,label=f"Train ROC (AUC={roc_auc_score(y_train, train_prob):.3f})")
        # Random line
        plt.plot([0, 1], [0, 1])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend()

        # Save
        plt.savefig(roc_path)

    def evaluate_model(self):

        # Load model bundle
        try:
            logger.info("Starting Model Evaluation Pipeline")
            bundle = joblib.load(self.model_artifact.model_path)
            logger.info("Loading trained model")
            model = bundle["model"]
            features = bundle["features"]
            
            
            X_train = pd.read_csv(self.bin_merge_artifact.X_train_final_path,usecols=features)
            y_train = pd.read_csv(self.fe_artifact.data_splits_y_train_path)
            y_train = y_train.values.ravel()
            
            train_prob = model.predict_proba(X_train)[:, 1]
            
            auc_train = roc_auc_score(y_train, train_prob)
            gini_train = self.calculate_gini(auc_train)
            ks_train = self.calculate_ks(y_train, train_prob)


            # Load test data
            X_test = pd.read_csv(self.bin_merge_artifact.X_test_final_path,usecols=features)
            y_test = pd.read_csv(self.fe_artifact.data_splits_y_test_path)
            y_test = y_test.values.ravel()
            

            # test metrics
            test_prob = model.predict_proba(X_test)[:, 1]
            
            auc_test = roc_auc_score(y_test, test_prob)
            gini_test = self.calculate_gini(auc_test)
            ks_test = self.calculate_ks(y_test, test_prob)

            # Save metrics
            metrics = {
                "AUC_train": float(auc_train),
                "GINI_train": float(gini_train),
                "KS_train": float(ks_train),

                "AUC_test": float(auc_test),
                "GINI_test": float(gini_test),
                "KS_test": float(ks_test)

            }
        

            with open(self.eval_artifact.metrics_path,"w") as f:
                
                json.dump(metrics,f,indent=4)
            logger.info( f"Metrics saved at:{self.eval_artifact.metrics_path}")
            
            roc_curve_path = self.eval_artifact.roc_curve_path
            self.save_roc_curve(y_train,train_prob,y_test,test_prob,roc_curve_path)        
            
            logger.info(metrics)
            logger.info("Model Evaluation Completed Successfully")
            return metrics
        except Exception as e:
            raise MyException(e,sys,logger)

if __name__ == "__main__":
    
    logger.info("Starting Model Evaluation Script")

    obj = ModelEvaluation()

    obj.evaluate_model()

    logger.info("Script execution finished")