import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD')
}

# MinIO configuration
MINIO_CONFIG = {
    'endpoint': os.getenv('MINIO_ENDPOINT'),
    'access_key': os.getenv('MINIO_ACCESS_KEY'),
    'secret_key': os.getenv('MINIO_SECRET_KEY'),
    'secure': False
}

# Feature configuration
FEATURES = ['feature1', 'feature2', 'feature1_minus_feature2', 'feature1_times_feature2']
TARGET_COLUMN = 'target'
PREDICTION_COLUMN = 'predictions'
MODEL_NAME_COLUMN = 'model_name'
DATE_COLUMN = 'date'
CUSTOMER_ID_COLUMN = 'customerId' 