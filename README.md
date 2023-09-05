# Active learning in BoTorch

## Installation
Create a new conda environment:
````
conda env create -f environment.yml
````

Install the project:
````
pip install -e .
````

## Running experiments
````
python <example_script>.py
````

### Known issues
1. For some reason, botorch optimization of acquisition function throws a grad NaN error because it ends up sampling only at the boundaries. This does not happen all the time, there is lot of randomness to this error. 
Some related issues are [#161](https://github.com/pytorch/botorch/issues/161), [#567](https://github.com/pytorch/botorch/issues/567)