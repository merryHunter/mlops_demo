# Port forward script for multiple services
$namespace = "ml-workflow"  # Change this if your deployment is in a different namespace

# Function to start port forwarding
function Start-PortForward {
    param (
        [string]$serviceName,
        [int]$localPort,
        [int]$remotePort
    )
    Write-Host "Starting port forward for $serviceName..."
    Write-Host "Access at: http://localhost:$localPort"
    Start-Process kubectl -ArgumentList "port-forward", "-n", $namespace, "svc/$serviceName", "$localPort`:$remotePort" -NoNewWindow
}

# Kill any existing port-forward processes
Get-Process kubectl -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*port-forward*" } | Stop-Process -Force

# Start port forwards for each service
Start-PortForward -serviceName "model-monitoring" -localPort 8501 -remotePort 80  # Streamlit UI
Start-PortForward -serviceName "minio" -localPort 9000 -remotePort 9000          # MinIO Console
Start-PortForward -serviceName "mlflow-service" -localPort 5000 -remotePort 5000 # MLflow UI

Write-Host "`nAll port forwards are running. Press Ctrl+C to stop all forwards."
Write-Host "Access the services at:"
Write-Host "- Model Monitoring: http://localhost:8501"
Write-Host "- MinIO Console: http://localhost:9000"
Write-Host "- MLflow UI: http://localhost:5000"

# Wait for user to press Ctrl+C
try {
    while ($true) { Start-Sleep -Seconds 1 }
}
finally {
    # Cleanup on exit
    Get-Process kubectl -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*port-forward*" } | Stop-Process -Force
} 