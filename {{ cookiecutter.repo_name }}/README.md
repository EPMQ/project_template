{{cookiecutter.project_name}}
==============================

{{cookiecutter.description}}

Project Organization
------------


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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
