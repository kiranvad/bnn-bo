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

If the above mode of installation does not work, please follow the instructions below:
1. create a conda environment using:
```bash
conda create -p /env/location/ python=3.11 -y
```

2. Install matplotlib, pandas, seaborn, openpyxl:
```bash
pip install matplotlib
pip install pandas openpyxl
pip install seaborn
```

3. Install botorch (that installs all the required pytroch packages):
```bash
pip install botorch
```

4. Install the `autophasemap` code to generate phase boundary predictions:
Follow instructions from : git clone https://github.com/pozzo-research-group/papers.git

## Running experiments
````
python <example_script>.py
````

### Known issues
1. For some reason, botorch optimization of acquisition function throws a grad NaN error because it ends up sampling only at the boundaries. This does not happen all the time, there is lot of randomness to this error. 
Some related issues are [#161](https://github.com/pytorch/botorch/issues/161), [#567](https://github.com/pytorch/botorch/issues/567)