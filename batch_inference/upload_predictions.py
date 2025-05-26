import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
import logging
from datetime import datetime
from minio import Minio
import io
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_HOST = os.getenv('POSTGRES_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'ml_monitoring')
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')

class MinioClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

    def read_parquet(self, bucket: str, object_name: str) -> pd.DataFrame:
        """Read parquet file from MinIO."""
        try:
            response = self.client.get_object(bucket, object_name)
            return pd.read_parquet(io.BytesIO(response.read()))
        except Exception as e:
            logger.error(f"Error reading parquet file {object_name}: {str(e)}")
            raise

def create_temp_table(cursor):
    """Create temporary table with the same schema as predictions."""
    cursor.execute("""
        CREATE TEMP TABLE temp_predictions (
            LIKE predictions INCLUDING ALL
        ) ON COMMIT DROP;
    """)

def bulk_load_from_csv(cursor, csv_file_path):
    """Bulk load data from CSV file using COPY command."""
    try:
        with open(csv_file_path, 'r') as f:
            # Skip header row
            next(f)
            cursor.copy_expert("""
                COPY temp_predictions (
                    date, customerId, feature1, feature2, target,
                    feature1_times_feature2, feature1_minus_feature2,
                    predictions, model_name
                ) FROM STDIN WITH (
                    FORMAT csv,
                    DELIMITER ','
                )
            """, f)
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise

def merge_tables(cursor):
    """Merge temporary table into main predictions table."""
    cursor.execute("""
        INSERT INTO predictions (
            id,
            date,
            customerId,
            feature1,
            feature2,
            feature1_minus_feature2,
            feature1_times_feature2,
            predictions,
            target,
            model_name,
            created_at
        )
        SELECT 
            nextval('predictions_id_seq'),
            date,
            customerId,
            feature1,
            feature2,
            feature1_minus_feature2,
            feature1_times_feature2,
            predictions,
            target,
            model_name,
            CURRENT_TIMESTAMP as created_at
        FROM temp_predictions
        ON CONFLICT (date, customerId, model_name) 
        DO UPDATE SET
            feature1 = EXCLUDED.feature1,
            feature2 = EXCLUDED.feature2,
            feature1_minus_feature2 = EXCLUDED.feature1_minus_feature2,
            feature1_times_feature2 = EXCLUDED.feature1_times_feature2,
            predictions = EXCLUDED.predictions,
            target = EXCLUDED.target;
    """)

def create_database():
    """Create the model_monitoring database if it doesn't exist."""
    try:
        # Connect to postgres database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname='postgres',
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'model_monitoring'")
            if not cursor.fetchone():
                logger.info("Creating model_monitoring database...")
                cursor.execute('CREATE DATABASE model_monitoring')
                logger.info("Database created successfully")
            else:
                logger.info("Database model_monitoring already exists")
        
        conn.close()
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        raise

def init_db_schema(cursor):
    """Initialize database schema if it doesn't exist."""
    # Check if predictions table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'predictions'
        );
    """)
    table_exists = cursor.fetchone()[0]
    
    if not table_exists:
        logger.info("Initializing database schema...")
        # Read and execute init.sql
        with open('init.sql', 'r') as f:
            init_script = f.read()
            # Split the script into individual statements
            statements = init_script.split(';')
            for statement in statements:
                if statement.strip():
                    try:
                        cursor.execute(statement)
                    except Exception as e:
                        logger.error(f"Error executing statement: {statement}")
                        logger.error(f"Error: {str(e)}")
                        raise
        logger.info("Database schema initialized successfully")

def upload_csv_to_pg(local_csv_fname: str) -> None:
    try:
        # Create database if it doesn't exist
        create_database()
        
        # Connect to model_monitoring database
        logger.info("Connecting to PostgreSQL...")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname='model_monitoring',
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        with conn.cursor() as cursor:
            # Initialize schema if needed
            init_db_schema(cursor)
            
            create_temp_table(cursor)
            
            # Bulk load from CSV
            logger.info("Bulk loading data from CSV...")
            bulk_load_from_csv(cursor, local_csv_fname)
            
            # Merge temporary table into main table
            logger.info("Merging tables...")
            merge_tables(cursor)
            
            # Commit the transaction
            conn.commit()
            
        logger.info("Successfully uploaded predictions to database!")
        
    except Exception as e:
        logger.error(f"Error uploading predictions: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
