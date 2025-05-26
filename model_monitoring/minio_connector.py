import pandas as pd
import io
from minio import Minio
from config import MINIO_CONFIG

class MinioConnector:
    def __init__(self):
        self.client = Minio(
            MINIO_CONFIG['endpoint'],
            access_key=MINIO_CONFIG['access_key'],
            secret_key=MINIO_CONFIG['secret_key'],
            secure=MINIO_CONFIG['secure']
        )
        self._cache = {}  # Cache for storing loaded data

    def _load_training_data(self, data_version: str, start_date, end_date):
        """Load all training data for a model into memory"""
        if data_version in self._cache:
            return self._cache[data_version]

        try:
            # Convert input dates to pandas datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            bucket_name = "train"
            dfs = []
            
            # List all objects recursively
            objects = self.client.list_objects(
                bucket_name,
                prefix=f"{data_version}/",
                recursive=True
            )
            
            # Load all CSV files
            for obj in objects:
                if obj.object_name.endswith('.parquet'):
                    try:
                        response = self.client.get_object(bucket_name, obj.object_name)
                        df = pd.read_parquet(io.BytesIO(response.read()))
                        df['date'] = pd.to_datetime(df['date'])
                        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                        df = df.loc[mask]
                        print(f"Loading chunk {obj.object_name}, len after filter: {df.shape[0]}")
                        if df.shape[0] > 0:
                            dfs.append(df)
                    except Exception as e:
                        print(f"Warning: Failed to read file {obj.object_name}: {str(e)}")
                        continue
            
            if not dfs:
                return pd.DataFrame()
            
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Cache the result
            self._cache[data_version] = combined_df
            return combined_df
            
        except Exception as e:
            raise Exception(f"Failed to load training data: {str(e)}")

    def get_training_data(self, data_version, start_date, end_date):
        """Fetch training data for a specific model and date range"""
        try:
            # Convert input dates to pandas datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Load all data for the model
            df = self._load_training_data(data_version, start_date, end_date)
            
            if df.empty:
                return df
            
            # Filter by date range
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            return df.loc[mask]
            
        except Exception as e:
            raise Exception(f"Failed to fetch training data: {str(e)}")

    def list_available_models(self):
        """List all models that have training data available"""
        try:
            bucket_name = "train"
            objects = self.client.list_objects(bucket_name, recursive=True)
            models = set()
            for obj in objects:
                # Get model name from first part of path
                model_name = obj.object_name.split('/')[0]
                if model_name:  # Skip empty strings
                    models.add(model_name)
            return list(models)
        except Exception as e:
            raise Exception(f"Failed to list available models: {str(e)}") 
