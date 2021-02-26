# Cookiecutter Template based on Cookiecutter-data-science for personal use

_A logical, reasonably standardized, but flexible project structure for doing and sharing data science work._


#### This project is based on [Cookiecutter-data-science](http://drivendata.github.io/cookiecutter-data-science/) and [Cookiecutter](https://cookiecutter.readthedocs.io/en/latest/)

This repository exists for anyone who wants to follow this scheme of directories.


### Requirements to use the cookiecutter template:
-----------
 - Python 2.7 or 3.5+
 - [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html) >= 1.4.0: This can be installed with pip by or conda depending on how you manage your Python packages:

``` bash
$ pip install cookiecutter
```

or

``` bash
$ conda config --add channels conda-forge
$ conda install cookiecutter
```


### To start a new project, run:
------------

    cookiecutter https://github.com/CcgEpmq/project_template.git



### The resulting directory structure
------------

The directory structure of your new project looks like this: 

```
├── LICENSE
├── README.md                  <- The top-level README for developers using this project.
├── dvc.yaml                   <- YAML file for DVC pipelines.
├── data                       <- Tree for data
│   ├── 01_raw                 <- Data from third party sources.
│   ├── 02_intermediate        <- Intermediate data that has been transformed.
│   ├── 03_primary             <- The final, canonical data sets for modeling.
│   ├── 04_model_input         <- Data to feed the models.
│   ├── 05_models              <- Model itself.
│   ├── 06_model_output        <- Results.
│   ├── 07_reporting           <- Reports.
│   └── 08_mlflow              <- Mlflow runs must be in this folder.
│
├── docs                       <- A default Sphinx project; see sphinx-doc.org for details
│
│
├── notebooks                  <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                 the creator's initials, and a short `-` delimited description, e.g.
│                                 `1.0_gc_initial_data_exploration`.
│
│
├── requirements.txt          <- The requirements file for reproducing the analysis environment, e.g.
│                                 generated with `pip freeze > requirements.txt`
│
└── src                       <- Source code [.py files].
    ├── 00_commons            <- Common scripts
    ├── 01_data_collection    <- Scripts for data collection (e.g API's) 
    ├── 02_data_preparation   <- Scripts for data preparation
    ├── 03_modelling          <- Scripts for modelling
    ├── 04_evaluation         <- Scripts for evaluation
    └── 05_reporting          <- Scripts to report results
```

