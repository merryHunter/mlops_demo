import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from db_connector import DatabaseConnector
from minio_connector import MinioConnector
from metrics import MetricsCalculator
from config import FEATURES, TARGET_COLUMN, PREDICTION_COLUMN
import os

# Initialize connections
db = DatabaseConnector()
minio = MinioConnector()
data_version = os.getenv('DATA_PREFIX')

# Page config
st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")
st.title("Model Monitoring Dashboard")

# Sidebar for model selection and date ranges
st.sidebar.header("Model Selection")
available_models = db.get_available_models()
selected_model = st.sidebar.selectbox("Select Model", available_models)

# Date range selection
st.sidebar.header("Date Range Selection")
performance_start = st.sidebar.date_input(
    "Performance Analysis Start Date",
    datetime.now() - timedelta(days=30)
)
performance_end = st.sidebar.date_input(
    "Performance Analysis End Date",
    datetime.now()
)

# take training data from 6 months ago
drift_start = st.sidebar.date_input(
    "Drift Analysis Start Date",
    datetime.now() - relativedelta(months=6)
)
drift_end = st.sidebar.date_input(
    "Drift Analysis End Date",
    datetime.now() - relativedelta(months=5)
)

def content():
    # Main content
    if selected_model:
        # Performance Metrics Section
        st.header("Model Performance Metrics")
        
        # Fetch predictions data
        predictions_df = db.get_model_predictions(
            selected_model,
            performance_start,
            performance_end
        )
        
        if not predictions_df.empty:
            # Calculate metrics
            metrics = MetricsCalculator.calculate_classification_metrics(
                predictions_df[TARGET_COLUMN],
                predictions_df[PREDICTION_COLUMN]
            )
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            col2.metric("Precision", f"{metrics['precision']:.3f}")
            col3.metric("Recall", f"{metrics['recall']:.3f}")
            col4.metric("F1 Score", f"{metrics['f1']:.3f}")
            
            # Plot daily performance
            daily_metrics = predictions_df.groupby('date').apply(
                lambda x: pd.Series(MetricsCalculator.calculate_classification_metrics(
                    x[TARGET_COLUMN], x[PREDICTION_COLUMN]
                ))
            ).reset_index()
            
            fig = px.line(daily_metrics, x='date', y=['accuracy', 'precision', 'recall', 'f1'],
                        title='Daily Performance Metrics')
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Drift Analysis Section
        st.header("Feature Drift Analysis")
        compute_feature_drift = st.button('Compute feature drift')
        
        if compute_feature_drift:
            try:
                # Fetch training data
                train_df = minio.get_training_data(data_version, drift_start, drift_end)
                
                if not train_df.empty and not predictions_df.empty:
                    # Calculate drift metrics
                    drift_metrics = MetricsCalculator.calculate_all_feature_drifts(
                        train_df, predictions_df, FEATURES
                    )
                    
                    # Display drift metrics
                    for feature in FEATURES:
                        st.subheader(f"Feature: {feature}")
                        metrics = drift_metrics[feature]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("KS Statistic", f"{metrics['ks_statistic']:.3f}")
                        col2.metric("P-value", f"{metrics['p_value']:.3f}")
                        col3.metric("Mean Drift", f"{metrics['mean_drift']:.3f}")
                        col4.metric("Std Drift", f"{metrics['std_drift']:.3f}")
                        
                        # Plot feature distributions
                        fig = px.histogram(
                            train_df,
                            x=feature,
                            title=f"{feature} Distribution (Training vs Prediction)",
                            color_discrete_sequence=['blue'],
                            opacity=0.7
                        )
                        fig.add_histogram(
                            x=predictions_df[feature],
                            name='Prediction',
                            opacity=0.7,
                            marker_color='red'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for drift analysis")
            
            except Exception as e:
                st.error(f"Error in drift analysis: {str(e)}")
    

if __name__ == "__main__":
    try:
        content()
    finally:
        # Cleanup
        db.close()