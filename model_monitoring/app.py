import streamlit as st
import duckdb
import pandas as pd
import os
from datetime import datetime, timedelta
import urllib.parse

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET_NAME = os.getenv("MINIO_BUCKET", "train")

def get_minio_url():
    """Generate MinIO S3-compatible URL"""
    # Parse the MinIO endpoint to get host and port
    parsed = urllib.parse.urlparse(MINIO_ENDPOINT)
    host = parsed.hostname
    port = parsed.port or 443
    protocol = 'https' if parsed.scheme == 'https' else 'http'
    
    return f"{protocol}://{host}:{port}"

def setup_duckdb_connection():
    """Setup DuckDB connection with HTTPFS extension and MinIO credentials"""
    con = duckdb.connect(database=':memory:')
    
    # Install and load HTTPFS extension
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    
    # Configure S3 credentials
    con.execute(f"""
        SET s3_access_key_id='{MINIO_ACCESS_KEY}';
        SET s3_secret_access_key='{MINIO_SECRET_KEY}';
        SET s3_endpoint='{MINIO_ENDPOINT}';
        SET s3_use_ssl= false;
        SET s3_url_style='path';
    """)
    
    return con

def get_parquet_files(date_range=None):
    """Get list of parquet files in the bucket using glob pattern"""
    con = setup_duckdb_connection()
    
    # Use glob pattern to list parquet files
    query = f"""
    SELECT *
    FROM read_parquet('s3://{BUCKET_NAME}/*.parquet')
    """
    
    return con.execute(query).df()

def main():
    st.title("Model Monitoring Dashboard")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range selector
    today = datetime.now()
    default_start = today - timedelta(days=7)
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(default_start, today),
        max_value=today
    )
    
    # Setup DuckDB connection
    con = setup_duckdb_connection()
    
    # Get list of parquet files
    files_df = get_parquet_files(date_range)
    
    if files_df.empty:
        st.warning("No data available for the selected date range")
        return
    
    # Model Performance Metrics
    st.header("Model Performance Metrics")
    
    # Accuracy over time - using direct S3 query
    accuracy_query = f"""
    SELECT 
        DATE(timestamp) as date,
        AVG(accuracy) as avg_accuracy
    FROM read_parquet('s3://{BUCKET_NAME}/results/{}*.parquet')
    WHERE DATE(timestamp) BETWEEN '{date_range[0].isoformat()}' AND '{date_range[1].isoformat()}'
    GROUP BY DATE(timestamp)
    ORDER BY date
    """
    
    accuracy_df = con.execute(accuracy_query).df()
    st.line_chart(accuracy_df.set_index('date'))
    
    # Feature Importance
    st.header("Feature Importance")
    feature_importance_query = f"""
    SELECT 
        feature_name,
        AVG(importance) as avg_importance
    FROM read_parquet('s3://{BUCKET_NAME}/*.parquet')
    WHERE DATE(timestamp) BETWEEN '{date_range[0].isoformat()}' AND '{date_range[1].isoformat()}'
    GROUP BY feature_name
    ORDER BY avg_importance DESC
    LIMIT 10
    """
    
    feature_importance_df = con.execute(feature_importance_query).df()
    st.bar_chart(feature_importance_df.set_index('feature_name'))
    
    # Data Quality Metrics
    st.header("Data Quality Metrics")
    quality_query = f"""
    SELECT 
        DATE(timestamp) as date,
        AVG(missing_values) as avg_missing,
        AVG(outliers) as avg_outliers
    FROM read_parquet('s3://{BUCKET_NAME}/*.parquet')
    WHERE DATE(timestamp) BETWEEN '{date_range[0].isoformat()}' AND '{date_range[1].isoformat()}'
    GROUP BY DATE(timestamp)
    ORDER BY date
    """
    
    quality_df = con.execute(quality_query).df()
    st.line_chart(quality_df.set_index('date'))
    
    # Raw Data View (with pagination)
    st.header("Raw Data")
    page_size = 1000
    total_rows_query = f"""
    SELECT COUNT(*) as count
    FROM read_parquet('s3://{BUCKET_NAME}/*.parquet')
    WHERE DATE(timestamp) BETWEEN '{date_range[0].isoformat()}' AND '{date_range[1].isoformat()}'
    """
    total_rows = con.execute(total_rows_query).fetchone()[0]
    
    num_pages = (total_rows + page_size - 1) // page_size
    page = st.sidebar.number_input("Page", 1, num_pages, 1)
    
    offset = (page - 1) * page_size
    raw_data_query = f"""
    SELECT *
    FROM read_parquet('s3://{BUCKET_NAME}/*.parquet')
    WHERE DATE(timestamp) BETWEEN '{date_range[0].isoformat()}' AND '{date_range[1].isoformat()}'
    LIMIT {page_size}
    OFFSET {offset}
    """
    
    raw_df = con.execute(raw_data_query).df()
    st.dataframe(raw_df)
    
    # Display pagination info
    st.sidebar.write(f"Showing rows {offset + 1} to {min(offset + page_size, total_rows)} of {total_rows}")

if __name__ == "__main__":
    main() 