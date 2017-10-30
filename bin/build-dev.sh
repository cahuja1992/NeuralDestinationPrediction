#!/bin/bash

# Build based docker container
docker build --build-arg="proxy=$http_proxy" -t destination-pred/python -f PythonDepDockerFile .

# Build add docker container
docker build  --build-arg="proxy=$http_proxy" -t destination-pred/app -f Dockerfile .
