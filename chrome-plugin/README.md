chrome-plugin
==============================

we make a chrome plugin which show the commits analysis in details and genrate a summary

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


create ec2 instance and also create secret key pair
now connect this instance to console and run thses command
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip -y
python3 -m venv mlflow_env
<!-- activate envirement -->
source mlflow_env/bin/activate
<!-- install the screen which automaticall start the mlfow server which any issue  -->
pip install mlflow boto3
sudo apt-get install screen -y
<!-- for starting the screen -->
screen -S mlfow

<!-- server run cammand -->
mlflow server --backend-store-uri ./mlruns --default-artifact-root s3://yt-chrome-plugin-bucket --host 0.0.0.0 --port 5000
<!-- for run server on browser -->
<!-- http://ec2-3-27-214-151.ap-southeast-2.compute.amazonaws.com:5000/
http://ec2-3-25-106-164.ap-southeast-2.compute.amazonaws.com/5000 -->
http://ec2-54-252-23-89.ap-southeast-2.compute.amazonaws.com:5000

<!-- to run in vscode intsall thiss -->
!pip install boto3
!pip install awscli
!aws configure
<!-- give follwing parameter -->
AWS Access Key ID [None]: 
AWS Secret Access Key [None]:
Default region name [None]: 
Default output format [None]: 