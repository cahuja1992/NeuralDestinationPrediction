# NEURAL DESTINATION PREDICTION

### 1. Build the container
`bin/build_dev.sh`
### 2. RUN the container
`docker run -v /home/admin1/work/P1-DestinationPrediction/git/NeuralDestinationPrediction/data:/data -it destination-pred/app /destpred.sh train mlp`
