# Classification Example using 1D Convnet with IMDB Dataset

From "Deep Learning with Python" by Francois Chollet, p. 249-254

## Dataset


## Model
Embedding + CNN_1D [MaxPool]

## Purpose
What should you learn:

### Workflow
* Functional workflow
* Implement CNN_1D layers
* Use callbacks and TensorBoard to monitor training

### Thoughtful DL
* Functional setup, devide pipeline into functional parts for better testability
* Implement callbacks to maintain control during training
    * EarlyStopping - stop training when accuracy stagnates
    * ModelCheckpoint - save best trained model automatically
* Use TensorBoard to monitor training
* Plot shape information of layers
* Model evaluation
