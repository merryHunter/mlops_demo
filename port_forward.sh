#!/bin/bash

# Port forward script for multiple services
NAMESPACE="ml-workflow"  # Change this if your deployment is in a different namespace

# Function to start port forwarding
start_port_forward() {
    local service_name=$1
    local local_port=$2
    local remote_port=$3
    echo "Starting port forward for $service_name..."
    echo "Access at: http://localhost:$local_port"
    kubectl port-forward -n $NAMESPACE svc/$service_name $local_port:$remote_port &
}

# Kill any existing port-forward processes
pkill -f "kubectl port-forward" || true

# Start port forwards for each service
start_port_forward "model-monitoring" 8501 80  # Streamlit UI
start_port_forward "minio" 9000 9000          # MinIO Console
start_port_forward "mlflow-service" 5000 5000 # MLflow UI

echo "All port forwards are running. Press Ctrl+C to stop all forwards."

# Wait for user to press Ctrl+C
wait

# Cleanup on exit
pkill -f "kubectl port-forward" 