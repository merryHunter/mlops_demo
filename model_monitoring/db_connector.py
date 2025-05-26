import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from config import DB_CONFIG

class DatabaseConnector:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
        except Exception as e:
            raise Exception(f"Failed to connect to database: {str(e)}")

    def get_model_predictions(self, model_name, start_date, end_date):
        """Fetch predictions data for a specific model and date range"""
        query = """
        SELECT date, customerId, feature1, feature2, feature1_minus_feature2,
                feature1_times_feature2, predictions, target, model_name
        FROM predictions
        WHERE model_name = %s
        AND date BETWEEN %s AND %s
        ORDER BY date
        """
        try:
            df = pd.read_sql_query(
                query,
                self.conn,
                params=(model_name, start_date, end_date)
            )
            return df
        except Exception as e:
            raise Exception(f"Failed to fetch predictions: {str(e)}")

    def get_available_models(self):
        """Get list of available model names"""
        query = "SELECT DISTINCT model_name FROM predictions"
        try:
            df = pd.read_sql_query(query, self.conn)
            return df['model_name'].tolist()
        except Exception as e:
            raise Exception(f"Failed to fetch model names: {str(e)}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close() 