#!/bin/sh
set -e -u
main() {
  if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <cluster-name>"
    echo "####Below are the kind cluster in you local machine.####"
    kind get clusters
    echo "########"
    exit 1
  fi
  echo "Starting to delete local Kubernetes cluster..."

  kind delete cluster --name "$1"

  echo "Successfully deleted local Kubernetes cluster."
}

# Execute the main function
main "$@"
