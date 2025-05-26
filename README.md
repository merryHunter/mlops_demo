# MLOps pipeline project Alameda

## Pre-requisites to run 

- Installed minikube and docker desktop
- disable firewall for network access if needed
- unzip data and mount location for data access

## Workflow

1. Assuming `data.zip` is unzipped to current directory `mlops_pipeline_demo`, start minikube with data mapping for data load:  
`minikube start --mount-string="D:/mlops_pipeline_demo/data:/data" --mount` or 
`minikube start --mount-string="/home/<user>/mlops_pipeline_demo/data:/data" --mount`

2. Create config map:  
`kubectl create configmap ml-config --from-file config-maps/config-map.yaml -n ml-workflow`  

3. Build docker containers for stages
`./build_docker_images.sh`

4. Start minio storage:
`kubectl apply -f deployments/minio.yaml -n=ml-workflow`  

5. Run `alameda_data_prep:25.05.3` container to load data from local path, apply feature engineering and store into MINIo.  
This step populates `train` and `predict` bucket with transformed features, and creates `holdout` bucket with 1st file from train set for model evaluation purpose.  
`kubectl apply -f deployments/data_prep.yaml`

6. Start MLFlow experiment tracking server:  
`kubectl apply -f deployments/mlflow.yaml`

7. Run `alameda_model_training:25.05.1`container:  
`kubectl apply -f deployments/model_training.yaml`

8. Run `alameda_batch_inference:25.05.1`container:  
`kubectl apply -f deployments/model_batch_inference.yaml`

9. Launch postgres for storing predictions:
`kubectl apply -f deployments/postgres.yaml`  

10. Start model monitoring with :  
`kubectl apply -f deployments/model_monitoring.yaml`

## Access UI dashboards

Run `./forward_ports.sh` and navigate to MINIo, MLFlow and AlamedaMonitoring dashboards.

## Limitations
- persistent storage option is not used to simplify installation process
- data memory usage overflow for non-distributed processing of data load for model training
- simplified data feature engineering (no map-reduce to efficiently calculate features across shards of data)
- config map is used for secrets for simplicity
- deployment specs are not fully parametrized

## Known issues

Reproducing fully might require manual involvement to properly set img tags for a consistent execution. 