import os
import io
import pickle
import pandas as pd
import numpy as np
from minio import Minio
from sklearn.base import BaseEstimator
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any
import logging
from datetime import datetime

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
            secure=True
        )

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

def process_shard(args: Dict[str, Any]) -> pd.DataFrame:
    """Process a single parquet shard."""
    minio_client = args['minio_client']
    bucket = args['bucket']
    object_name = args['object_name']
    model = args['model']
    features = args['features']

    try:
        # Read parquet file
        df = minio_client.read_parquet(bucket, object_name)
        
        # Make predictions
        predictions = model.predict(df[features])
        
        # Add predictions to dataframe
        df['predictions'] = predictions
        
        return df
    except Exception as e:
        logger.error(f"Error processing shard {object_name}: {str(e)}")
        raise

def main():
    # MinIO configuration
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    MODEL_BUCKET = os.getenv('MODEL_BUCKET', 'models')
    DATA_BUCKET = os.getenv('DATA_BUCKET', 'data')
    RESULTS_BUCKET = os.getenv('RESULTS_BUCKET', 'results')
    MODEL_PATH = os.getenv('MODEL_PATH', 'production/model.pkl')
    DATA_PREFIX = os.getenv('DATA_PREFIX', 'input/')
    FEATURES = os.getenv('FEATURES', '').split(',')
    
    if not all([MINIO_ACCESS_KEY, MINIO_SECRET_KEY]):
        raise ValueError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set")

    # Initialize MinIO client
    minio_client = MinioClient(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)
    
    # Load model
    logger.info("Loading model from MinIO...")
    model = minio_client.get_model(MODEL_BUCKET, MODEL_PATH)
    
    # Get list of parquet files
    logger.info("Listing parquet files...")
    parquet_files = minio_client.get_parquet_shards(DATA_BUCKET, DATA_PREFIX)
    
    if not parquet_files:
        logger.warning("No parquet files found!")
        return
    
    # Prepare arguments for multiprocessing
    process_args = [{
        'minio_client': minio_client,
        'bucket': DATA_BUCKET,
        'object_name': file,
        'model': model,
        'features': FEATURES
    } for file in parquet_files]
    
    # Process files in parallel
    logger.info(f"Processing {len(parquet_files)} files using {cpu_count()} processes...")
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_shard, process_args)
    
    # Combine results
    final_df = pd.concat(results, ignore_index=True)
    
    # Generate timestamp-based prefix
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_prefix = f"{timestamp}/"
    output_filename = "predictions.parquet"
    output_path = f"{output_prefix}{output_filename}"
    
    # Save results to MinIO
    logger.info(f"Saving predictions to MinIO bucket '{RESULTS_BUCKET}' with prefix '{output_prefix}'...")
    minio_client.save_parquet(final_df, RESULTS_BUCKET, output_path)
    
    logger.info("Batch inference completed successfully!")

if __name__ == "__main__":
    main() 