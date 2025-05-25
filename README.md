# MLOps pipeline project Alameda

# Pre-requisites to run 

- Installed minikube and docker desktop
- disable firewall for network access if needed

# Workflow:

1. Start minikube with data mapping for data load:
` minikube start --mount-string="$HOME/go/src/github.com/nginx:/data" --mount`
2. Create config map:
`kubectl create configmap ml-config --from-file config-maps/config-map.yaml -n ml-workflow`
3. Build docker containers for stages
`./build_docker_images.sh` (?)
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
`kubectl apply -f deployments/batch_inference.yaml`
9. Start model monitoring with duckdb for data querying:
`kubectl apply -f deployments/batch_inference.yaml`

# Limitations
- persistent storage option is not used to simplify installation process
- data memory usage overflow for non-distributed processing of data load for model training
- simplified data feature engineering (no map-reduce to efficiently calculate features across shards of data)

## todo:

- run batch inference
- register / tag best model with tag 'production'
- retrieve model for batch inference with tag 'production'
- write model informaiton alongside results predict