# MLOps pipeline project Alameda

# Pre-requisites to run 

1. Installed minikube and docker

Environment used for development:

Workflow:
1. Start minikube with data mapping for data load:
` minikube start --mount-string="$HOME/go/src/github.com/nginx:/data" --mount`
2. Build docker containers for stages
`./build_docker_images.sh` (?)
3. Run `alameda_data_prep:25.05.3` container to load data from local path, apply feature engineering and store into MINIo.
This step populates `train` and `predict` bucket with transformed features, and creates `holdout` bucket with 1st file from train set for model evaluation purpose.
`kubectl apply -f deployments/data_prep.yaml`
4. Start MLFlow experiment tracking server:
`kubectl apply -f config-maps/mlflow-configmap.yaml`
`kubectl apply -f deployments/mlflow.yaml`
5. Run `alameda_model_training:25.05.1`container:
``

# Limitations
- persistent storage option is not used to simplify installation process
- data memory usage overflow for non-distributed processing of data load for model training
- simplified data feature engineering (no map-reduce to efficiently calculate features across shards of data)

## Issues:

- configmap values?


commands:


`kubectl create configmap ml-config --from-file .\config-map.yaml -n ml-workflow`
``