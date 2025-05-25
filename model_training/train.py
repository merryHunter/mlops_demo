import os
import mlflow
import numpy as np
import pandas as pd
from minio import Minio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                        f1_score, roc_auc_score, confusion_matrix
import pyarrow.parquet as pq
import io
import psutil
from datetime import datetime
import logging
from typing import Dict
import pickle
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomerClassifierTrainer:
    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        bucket_name: str = "train",
        data_version: str = "d1",
        model_version: str = "v1",
        test_size: float = 0.2,
        random_state: int = 42,
        max_iter: int = 1000,
        mlflow_uri: str = ""
    ):
        """Initialize the trainer with configuration parameters"""
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.bucket_name = bucket_name
        self.data_version = data_version
        self.model_version = model_version
        self.test_size = test_size
        self.random_state = random_state
        self.max_iter = max_iter

        self.minio_client = None
        self.model = None
        self.metrics = None
        self.feature_importance = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"clf_m-{self.model_version}_d-{self.data_version}_t-{self.timestamp}"

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_uri)

    def _get_minio_client(self) -> Minio:
        """Initialize and return MinIO client"""
        if self.minio_client is None:
            self.minio_client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=False
            )
        return self.minio_client

    def _get_available_memory(self) -> float:
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024 * 1024 * 1024)

    def _estimate_dataframe_memory(self, df: pd.DataFrame) -> float:
        """Estimate memory usage of a pandas DataFrame in GB"""
        memory_usage = df.memory_usage(deep=True).sum()
        return memory_usage / (1024 * 1024 * 1024)  # Convert to GB

    def load_data(self, bucket_name, prefix) -> pd.DataFrame:
        """Load data from MinIO with memory constraints"""
        minio_client = self._get_minio_client()
        
        # List all parquet files in the bucket with the given prefix.
        logger.info(f"Loading data from {bucket_name}, prefix: {prefix}")
        objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
        parquet_files = [obj.object_name for obj in objects]
        logger.info(f"Found {len(parquet_files)} files")
        
        if not parquet_files:
            raise ValueError(f"No files found in bucket {bucket_name} with prefix {prefix}")
        
        # First, load a single file to estimate memory usage
        first_file = parquet_files[0]
        logger.info(f"Loading first file {first_file} to estimate memory usage")
        response = minio_client.get_object(bucket_name, first_file)
        sample_df = pd.read_parquet(io.BytesIO(response.read()))
        single_file_memory = self._estimate_dataframe_memory(sample_df)
        
        # Estimate total memory needed
        estimated_total_memory = single_file_memory * len(parquet_files)
        logger.info(f"Estimated memory per file: {single_file_memory:.2f} GB")
        logger.info(f"Estimated total memory needed: {estimated_total_memory:.2f} GB")
        
        # Load all files
        dfs = [sample_df]  # Start with the already loaded first file
        for file_name in parquet_files[1:]:  # Skip the first file as it's already loaded
            logger.info(f"Loading file {file_name}")
            response = minio_client.get_object(bucket_name, file_name)
            df = pd.read_parquet(io.BytesIO(response.read()))
            dfs.append(df)
        
        final_df = pd.concat(dfs, ignore_index=True)
        actual_memory = self._estimate_dataframe_memory(final_df)
        
        logger.info(f"Loaded {len(parquet_files)} files with {len(final_df)} rows")
        logger.info(f"Actual memory usage: {actual_memory:.2f} GB")
        logger.info(final_df.columns)
        return final_df

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """Train logistic regression model"""
        model = LogisticRegression(
            max_iter=self.max_iter,
            class_weight='balanced',
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.model = model
        return model

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance for binary classification"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        self.metrics = metrics
        return metrics

    def log_to_mlflow(self, holdout_ratio: float, cols: list) -> None:
        """Log all metrics and parameters to MLflow"""
        # Log parameters
        mlflow.log_param("model_version", self.model_version)
        mlflow.log_param("data_version", self.data_version)
        mlflow.log_param("test_size", self.test_size)
        mlflow.log_param("random_state", self.random_state)
        mlflow.log_param("max_iter", self.max_iter)
        mlflow.log_param("holodut_target_ratio", holdout_ratio)

        # Log metrics
        for metric_name, metric_value in self.metrics.items():
            if metric_name != 'confusion_matrix':
                mlflow.log_metric(metric_name, metric_value)
        
        # Log confusion matrix
        conf_matrix_df = pd.DataFrame(
            self.metrics['confusion_matrix'],
            columns=['Predicted Negative', 'Predicted Positive'],
            index=['Actual Negative', 'Actual Positive']
        )
        mlflow.log_table(conf_matrix_df, "confusion_matrix.json")
        
        # Log model
        mlflow.sklearn.log_model(self.model, self.run_name)
        
        # Log feature importance
        self.feature_importance = pd.DataFrame({
            'feature': cols,
            'importance': np.abs(self.model.coef_[0])
        })
        mlflow.log_table(self.feature_importance, "feature_importance.json")

    def save_model_to_minio(self, run_id: str) -> None:
        """Save model to MinIO in pickle format"""
        try:
            # Create pickle buffer
            model_buffer = io.BytesIO()
            pickle.dump(self.model, model_buffer)
            model_buffer.seek(0)
            
            # Create model path with run ID
            model_path = f"{self.run_name}/model.pkl"
            
            # Upload to MinIO
            minio_client = self._get_minio_client()
            
            # Ensure models bucket exists
            if not minio_client.bucket_exists("models"):
                minio_client.make_bucket("models")
                logger.info("Created 'models' bucket")
            
            # Upload model
            minio_client.put_object(
                bucket_name="models",
                object_name=model_path,
                data=model_buffer,
                length=model_buffer.getbuffer().nbytes,
                content_type="application/octet-stream"
            )
            
            logger.info(f"Model saved to MinIO: models/{model_path}")
            
            # Log model path to MLflow
            mlflow.log_param("model_path", f"models/{model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model to MinIO: {e}")
            raise

    def train(self) -> None:
        """Main training pipeline"""
        # Load and prepare data
        df_train = self.load_data(bucket_name="train", prefix=f"{self.data_version}")
        y_train = df_train['target'].copy()
        X_train = df_train.drop(['date', 'customerId', 'target'], axis=1)
        df_holdout = self.load_data(bucket_name="holdout", prefix=f"{self.data_version}")
        # Verify binary classification
        unique_classes = np.unique(y_train)
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification, but found {len(unique_classes)} classes: {unique_classes}")
        
        y_test = df_holdout['target']
        X_test = df_holdout.drop(['date', 'customerId', 'target'], axis=1)
        
        with mlflow.start_run(run_name=self.run_name) as run:
            try:
                # Train and evaluate
                self.train_model(X_train, y_train)
                self.evaluate_model(X_test, y_test)
                target_ratio = sum(y_test) / len(y_test)
                
                # Log everything to MLflow
                self.log_to_mlflow(target_ratio, list(X_train.columns))
                
                # Save model to MinIO
                self.save_model_to_minio(run.info.run_id)
                
                # Log results
                logger.info("\nModel Performance Metrics:")
                logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
                logger.info(f"Precision: {self.metrics['precision']:.4f}")
                logger.info(f"Recall: {self.metrics['recall']:.4f}")
                logger.info(f"F1 Score: {self.metrics['f1']:.4f}")
                logger.info(f"ROC AUC: {self.metrics['roc_auc']:.4f}")
                logger.info("\nConfusion Matrix:")
                logger.info(pd.DataFrame(
                    self.metrics['confusion_matrix'],
                    columns=['Predicted Negative', 'Predicted Positive'],
                    index=['Actual Negative', 'Actual Positive']
                ))
                logger.info(f"\nModel logged to MLflow run: {run.info.run_id}")
                
            except Exception as e:
                logger.error(f"Error during training: {e}")
                raise

def main():
    trainer = CustomerClassifierTrainer(
        minio_endpoint=os.getenv("MINIO_ENDPOINT"),
        minio_access_key=os.getenv("MINIO_ACCESS_KEY"),
        minio_secret_key=os.getenv("MINIO_SECRET_KEY"),
        data_version=os.getenv("DATA_PREP_TAG"),
        model_version=os.getenv("IMG_TAG"),
        mlflow_uri=os.getenv("MLFLOW_TRACKING_URI")
    )
    trainer.train()

if __name__ == "__main__":
    main() 
    sys.exit(0)