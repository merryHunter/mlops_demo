# MLOps pipeline project Alameda




## Issues:

- configmap values?


commands:

` minikube start --mount-string="$HOME/go/src/github.com/nginx:/data" --mount`
`kubectl create configmap ml-config --from-file .\config-map.yaml -n ml-workflow`
``