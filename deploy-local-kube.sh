#!/bin/sh
set -e -u

create_flask_app_image() {
  echo "Building flask app image..."
  docker build -t cancellation-prediction:v1  -f  ./app/cancellation_prediction_app_image.dockerfile .
  echo "Built flask app image."
}

create_kind_cluster() {
  local name="$1"
  echo "Creating local Kubernetes cluster... $name"
  kind create cluster --name $name
  echo "Local Kubernetes cluster created with name $name."
}

load_images_to_kind_cluster() {
  local name="$1"
  echo "Loading images to Kind repository..."
  kind load docker-image cancellation-prediction:v1 --name $name
  echo "Images loaded to Kind repository."
}

deploy_flask_app() {
  echo "Deploying gateway..."
  kubectl apply -f ./kube-config/flask-app-deployments.yaml
  kubectl apply -f ./kube-config/flask-app-service.yaml
  echo "Gateway deployed."
}

main() {
  if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <cluster-name>"
    exit 1
  fi
  echo "Starting deployment to local Kubernetes cluster..."

  create_flask_app_image

  create_kind_cluster "$1"
  load_images_to_kind_cluster "$1"

  deploy_flask_app

  echo "Successfully deployed to local Kubernetes cluster. Please port forward and test the deployment."
}

# Execute the main function
main "$@"
