import os
import polars as pl
from datetime import datetime
from pathlib import Path
from minio import Minio
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get MinIO configuration from environment variables
minio_endpoint = os.getenv('MINIO_ENDPOINT')
if not minio_endpoint:
    raise ValueError("MINIO_ENDPOINT environment variable is not set")

NFS_MOUNT_PATH =  os.getenv("NFS_MOUNT_PATH")
minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
minio_secure = os.getenv('MINIO_SECURE', 'False').lower() == 'true'

# Version prefix to have lineage with docker image version
VERSION_PREFIX = os.getenv("IMG_TAG") 
CREATE_HOLDOUT = os.getenv("CREATE_HOLDOUT")

# Constants
BUCKETS = ["train", "predict"]
HOLDOUT_BUCKET = "holdout"

class MinioETL:
    def __init__(self):
        self.client = Minio(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure
        )
        self._ensure_buckets_exist()

    def _ensure_buckets_exist(self):
        """Create MinIO buckets if they don't exist."""
        for bucket in BUCKETS:
            if not self.client.bucket_exists(bucket):
                logger.info(f"Creating bucket: {bucket}")
                self.client.make_bucket(bucket)

    def _get_partition_path(self, df: pl.DataFrame, partition_cols: List[str]) -> str:
        """Generate partition path based on partition columns."""
        partition_values = []
        for col in partition_cols:
            if col in df.columns:
                partition_values.append(f"{col}={df[col].iloc[0]}")
        return "/".join(partition_values)

    def prepare_data_chunk(self, df: pl.DataFrame) -> pl.DataFrame:
        """Build features"""
        # multiply feature1 and feature2, take difference
        df = df.with_columns([
            (df["feature1"] * df["feature2"]).alias("feature1_times_feature2"),
            (df["feature1"] - df["feature2"]).alias("feature1_minus_feature2"),
        ])
        return df

    def process_directory(self, source_dir: str, target_bucket: str) -> None:
        """Process all parquet files in a directory."""
        source_path = Path(source_dir)
        if not source_path.exists():
            raise ValueError(f"Source directory {source_dir} does not exist")

        parquet_files = list(source_path.glob("**/*.parquet"))
        logger.info(f"Found {len(parquet_files)} parquet files in {source_dir}")
        temp_filename = "temp.parquet"
        for file_path in parquet_files:
            try:
                # Read parquet file
                df = pl.read_parquet(file_path)
                
                # Construct object name with versioning and partitioning
                object_name = f"{VERSION_PREFIX}/{file_path.name}"
                df = self.prepare_data_chunk(df)

                # Convert DataFrame back to parquet and upload to MinIO
                df.write_parquet(temp_filename)
                self.client.fput_object(
                    bucket_name=target_bucket,
                    object_name=object_name,
                    file_path=temp_filename,
                )
                
                logger.info(f"Successfully uploaded {file_path.name} to {target_bucket}/{object_name}")

            except Exception as e:
                logger.error(f"Failed to upload {file_path.name} to {target_bucket}/{object_name}: {str(e)}")

    def create_holdout_set(self, train_bucket, prefix, holdout_bucket) -> None:
        """ 
        Selects first file in train bucket and moves it to holdout bucket.
        holdout_size - number of samples
        args: target_ratio - distribution of 1/0 in holdout
        """
        if not self.client.bucket_exists(holdout_bucket):
            logger.info(f"Creating bucket: {holdout_bucket}")
            self.client.make_bucket(holdout_bucket)
        holdout_object_name = f"{prefix}.parquet"
        # get all objects in train and get 1
        for obj in self.client.list_objects(train_bucket, recursive=True):
            if obj.object_name.endswith('.parquet'):
                train_object_name = obj.object_name
                break

        temp_filename = "temp.parquet"
        self.client.fget_object(
            bucket_name=train_bucket,
            object_name=train_object_name,
            file_path=temp_filename,
        )
        self.client.fput_object(
            bucket_name=HOLDOUT_BUCKET,
            object_name=holdout_object_name,
            file_path=temp_filename,
        )
        # remove file from train bucket
        self.client.remove_object(
            bucket_name=train_bucket,
            object_name=train_object_name,
        )
        return None
    

def main():
    try:
        etl = MinioETL()
        
        for bucket in BUCKETS:
            source_dir = os.path.join(NFS_MOUNT_PATH, bucket)
            if os.path.exists(source_dir):
                logger.info(f"Processing data for {bucket} bucket")
                etl.process_directory(source_dir, bucket)
            else:
                logger.warning(f"Source directory {source_dir} does not exist, skipping {bucket}")

        if CREATE_HOLDOUT == "true":
            etl.create_holdout_set(train_bucket="train", 
                                   prefix=VERSION_PREFIX,
                                    holdout_bucket=HOLDOUT_BUCKET)

            
    except Exception as e:
        logger.error(f"ETL process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

