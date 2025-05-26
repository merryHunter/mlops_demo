import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ks_2samp

class MetricsCalculator:
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred):
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

    @staticmethod
    def calculate_feature_drift(train_data, predict_data, feature):
        """Calculate drift metrics for a single feature"""
        # Calculate KS test statistic and p-value
        ks_statistic, p_value = ks_2samp(
            train_data[feature].dropna(),
            predict_data[feature].dropna()
        )
        
        # Calculate distribution statistics
        train_mean = train_data[feature].mean()
        predict_mean = predict_data[feature].mean()
        train_std = train_data[feature].std()
        predict_std = predict_data[feature].std()
        
        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'train_mean': train_mean,
            'predict_mean': predict_mean,
            'train_std': train_std,
            'predict_std': predict_std,
            'mean_drift': abs(train_mean - predict_mean),
            'std_drift': abs(train_std - predict_std)
        }

    @staticmethod
    def calculate_all_feature_drifts(train_data, predict_data, features):
        """Calculate drift metrics for all features"""
        drift_metrics = {}
        for feature in features:
            drift_metrics[feature] = MetricsCalculator.calculate_feature_drift(
                train_data, predict_data, feature
            )
        return drift_metrics 