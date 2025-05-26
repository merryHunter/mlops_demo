import os
import io
import pickle
import pandas as pd
from minio import Minio
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any
import logging
from datetime import datetime
from upload_predictions import upload_csv_to_pg


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinioClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

    def ensure_buckets_exist(self, bucket):
        """Create MinIO buckets if they don't exist."""
        if not self.client.bucket_exists(bucket):
            logger.info(f"Creating bucket: {bucket}")
            self.client.make_bucket(bucket)

    def get_model(self, bucket: str, model_path: str) -> BaseEstimator:
        """Load model from MinIO."""
        try:
            response = self.client.get_object(bucket, model_path)
            model = pickle.loads(response.read())
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_parquet_shards(self, bucket: str, prefix: str) -> List[str]:
        """List all parquet files in the specified prefix."""
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects if obj.object_name.endswith('.parquet')]
        except Exception as e:
            logger.error(f"Error listing parquet files: {str(e)}")
            raise

    def read_parquet(self, bucket: str, object_name: str) -> pd.DataFrame:
        """Read parquet file from MinIO."""
        try:
            response = self.client.get_object(bucket, object_name)
            return pd.read_parquet(io.BytesIO(response.read()))
        except Exception as e:
            logger.error(f"Error reading parquet file {object_name}: {str(e)}")
            raise

    def save_parquet(self, df: pd.DataFrame, bucket: str, object_name: str) -> None:
        """Save DataFrame as parquet file to MinIO."""
        try:
            # Convert DataFrame to parquet bytes
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)
            
            # Upload to MinIO
            self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=parquet_buffer,
                length=parquet_buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )
            logger.info(f"Successfully saved predictions to {bucket}/{object_name}")
        except Exception as e:
            logger.error(f"Error saving predictions to MinIO: {str(e)}")
            raise

    def get_latest_model(self, bucket: str):
        """Get the latest model pickle file from MinIO based on timestamp."""
        try:
            # List all objects in the bucket with the given prefix
            objects = self.client.list_objects(bucket, recursive=True)
            
            # Filter for .pkl files and sort by last modified time
            model_files = [
                obj for obj in objects 
                if obj.object_name.endswith('.pkl')
            ]
            
            if not model_files:
                raise ValueError(f"No model files found in {bucket}")
            
            # Sort by last modified time (newest first)
            latest_model = sorted(model_files, key=lambda x: x.last_modified, reverse=True)[0]
            
            # Download and load the model
            response = self.client.get_object(bucket, latest_model.object_name)
            model_data = response.read()
            
            # Load the model as a logistic regression
            model = pickle.loads(model_data)
            if not isinstance(model, LogisticRegression):
                raise ValueError("Loaded model is not a LogisticRegression instance")
            
            logger.info(f"Successfully loaded latest model from {latest_model.object_name}")
            return model, latest_model.object_name.split('/')[-2] # skip pickle
            
        except Exception as e:
            logger.error(f"Error loading latest model from MinIO: {str(e)}")
            raise

def process_shard(args: Dict[str, Any]) -> pd.DataFrame:
    """Process a single parquet shard."""
    bucket = args['bucket']
    object_name = args['object_name']
    model = args['model']
    minio_config = args['minio_config']
    model_name = args['model_name']

    try:
        # Create new MinioClient instance for this process
        minio_client = MinioClient(
            endpoint=minio_config['endpoint'],
            access_key=minio_config['access_key'],
            secret_key=minio_config['secret_key']
        )
        
        # Read parquet file
        logger.info(f"Reading {bucket}/{object_name} ...")
        response = minio_client.client.get_object(bucket, object_name)
        df = pd.read_parquet(io.BytesIO(response.read()))
        
        # Make predictions
        predictions = model.predict(df.drop(['date', 'customerId', 'target'], axis=1))
        
        # Add predictions to dataframe
        df['predictions'] = predictions
        df['model_name'] = model_name
        
        return df
    except Exception as e:
        logger.error(f"Error processing shard {object_name}: {str(e)}")
        raise

def main():
    # MinIO configuration
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    DATA_BUCKET = os.getenv('DATA_BUCKET', 'predict')
    RESULTS_BUCKET = os.getenv('RESULTS_BUCKET', 'results')
    PREDICT_PREFIX = os.getenv('PREDICT_PREFIX')
    
    if not all([MINIO_ACCESS_KEY, MINIO_SECRET_KEY]):
        raise ValueError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set")

    # Initialize MinIO client
    minio_client = MinioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)
    
    logger.info("Loading model from Minio...")
    model, model_name = minio_client.get_latest_model(
            bucket=os.getenv('MODEL_BUCKET', 'models')
    )
    
    # Get list of parquet files
    logger.info("Listing parquet files...")
    parquet_files = minio_client.get_parquet_shards(DATA_BUCKET, PREDICT_PREFIX)
    
    if not parquet_files:
        logger.warning("No parquet files found!")
        return
    
    # Prepare arguments for multiprocessing
    minio_config = {
        'endpoint': MINIO_ENDPOINT,
        'access_key': MINIO_ACCESS_KEY,
        'secret_key': MINIO_SECRET_KEY
    }
    
    process_args = [{
        'bucket': DATA_BUCKET,
        'object_name': file,
        'model': model,
        'minio_config': minio_config,
        'model_name': model_name
    } for file in parquet_files]
    
    # Process files in parallel
    n_processes = min(cpu_count(), len(process_args))
    logger.info(f"Processing {len(parquet_files)} files using {n_processes} processes...")
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_shard, process_args)
    
    # Combine results
    final_df = pd.concat(results, ignore_index=True)
    
    # Generate timestamp-based prefix
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_prefix = f"{PREDICT_PREFIX}/{model_name}/{timestamp}"
    output_filename = "predictions.parquet"
    output_path = f"{output_prefix}/{output_filename}"
    
    # Save results to MinIO
    logger.info(f"Saving predictions to MinIO bucket '{RESULTS_BUCKET}' with prefix '{output_prefix}'...")
    minio_client.ensure_buckets_exist(RESULTS_BUCKET)
    minio_client.save_parquet(final_df, RESULTS_BUCKET, output_path)
    local_fname = 'predictions.csv'
    final_df.to_csv(local_fname, index=False)
    upload_csv_to_pg(local_fname)
    logger.info("Batch inference completed successfully!")

if __name__ == "__main__":
    main() 