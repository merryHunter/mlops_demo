#!/bin/bash

cd data_prep/
minikube image build -t alameda_data_prep:25.05.3 .
cd ../model_training/
minikube image build -t alameda_model_training:25.05.3 .
cd ../model_monitoring/
minikube image build -t alameda_model_monitoring:25.05.1 .