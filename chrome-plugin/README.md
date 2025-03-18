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

tep-by-Step Guide: Configuring DVC with AWS S3 as Remote Storage
1️⃣ Create an S3 Bucket
Go to AWS Console → S3 → Click Create bucket.
Choose a unique name (e.g., campusxproject2bucket).
Select a region and keep default settings.
Click Create bucket.
2️⃣ Create an IAM User (Skip if Already Exists)
Go to AWS IAM Console → Users → Click Add users.
Enter a username and enable Programmatic access.
Attach "AmazonS3FullAccess" policy.
Create and download the access keys (you’ll need them later).
3️⃣ Install Required Packages
Run the following command in your terminal:


pip install dvc[s3] awscli
dvc[s3]: Adds S3 support to DVC.
awscli: AWS Command Line Interface for authentication.
4️⃣ Remove Existing DVC Remote (If Any)
Run:

dvc remote remove myremote
This removes any previous remote named myremote.
5️⃣ Configure AWS CLI for Authentication
Run:


aws configure
Enter the AWS Access Key and Secret Key from Step 2.
Set region (e.g., us-east-1).
Keep output format as default (press Enter).
6️⃣ Add an S3 Remote to DVC
Run:


dvc remote add -d myremote s3://campusxproject2bucket
myremote: Name of the remote storage.
-d: Sets it as the default DVC remote.
s3://campusxproject2bucket: Path to the S3 bucket.
7️⃣ Add Changes to Git
Run:


git add .
This stages all changes for commit.
8️⃣ Commit Changes to Git
Run:

git commit -m "Configured DVC with S3 remote"
Saves the changes in Git history.
9️⃣ Push Data to S3 Using DVC
Run:

dvc push
Uploads tracked files to S3.
🔟 Push Code to GitHub

git push origin main
Pushes the code repository to GitHub.
✅ Final Check
Run dvc remote list to verify the remote is added.
Run dvc status to check if any changes need to be pushed.
Now, your DVC is configured with AWS S3 as remote storage. 🚀

s3 bucket:
  Amazon Resource Name (ARN)
arn:aws:s3:::yt-chrome-plugin-bucket


If you want to install DVC with S3 support, you should use the following command:


pip install dvc[s3]

dvc remote add -d myremote s3://yt-chrome-plugin-bucket