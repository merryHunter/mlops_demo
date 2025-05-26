#!/bin/bash

# Set default tag if not provided
IMAGE_TAG=${IMAGE_TAG:-"25.05.1"}

cd data_prep/
minikube image build -t alameda_data_prep:${IMAGE_TAG} .
cd ../model_training/
minikube image build -t alameda_model_training:${IMAGE_TAG} .
cd ../model_monitoring/
minikube image build -t alameda_model_monitoring:${IMAGE_TAG} .