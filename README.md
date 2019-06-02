
<!-- tocstop -->

QA partner: Tian Fu

## Project Charter 

**Vision**: Banks usually struggle to target customers with insufficient or non-existent credit histories and have a hard time to decide whether to grant loans to those customers. This project will help the bank industry evaluate the repayment ability of those “underserved” customers, expand the loan servicing and generate more revenue.

**Mission**: Profiling of customers who are likely to default will be provided to help the bank filter out high-risk customers in the first stage. The bank will also be able to classify a new customer into the category of “with payment difficult” and “without payment difficult” to decide if they want to grant loans to the customer. The dataset used to train the classification model contains personal background, previous credits and loan application with other institutions of approximately 300,000 applicants. 

**Success criteria**: 1). AUC, Sensitivity and Specificity of test data will be applied to measure the performance of the model. Considering the available models in the market whose AUCs are between 0.65-0.73, we will set the minimum value of test AUC as 0.70 for success. 2). Whether the additional revenue from lending to those underserved customers reaches the finance goal of the bank (e.g. less than 2% default risk, generating more than 1 million revenue from those customers) will be used to evaluate the business outcome.

## Backlog
**Themes**: There are distinct types of unbanked customers. Some of them have repayment capability and can contribute to the reliable source of income to the bank, and therefore not providing loan servicing for all of them will be a loss to the bank. Through this project with the customer profiling and classification model, the bank will be able to identify high-quality customers with insufficient credit history. The bank can establish positive and safe loaning relationship with them, generate higher revenue and have more business opportunity. 

o **Epic1**: Explore various types of customers unserved by the bank from different aspects to identify what type of customers are likely to have late payment 

♣ *Backlog1*: Compare the customers with late payment and without late payment based on their individual background data (e.g. income, family, housing, education) –2 points (planned for the next two weeks)

♣ *Backlog2*: Visualize the credit behavior from Credit Bureau of late-payment population and non-late-payment population –4 points (planned for the next two weeks)

♣ *Backlog3*: Analyze the previous loan applications of the customers and study the correlation between previous loan application situation and late-payment behavior of future loans –4 points 

♣ *Backlog4*: Compare the customers with late payment and without late payment based on their social surrounding data (e.g. how many observations of the applicant social surrounding defaulted) –2 points

o **Epic2**: Based on their background, credit and transaction information in other financial institutions, classify customers into two groups: with payment difficulty and without payment difficulty 

♣ *Backlog1*: Understand, clean and merge different datasets including applicant background data, bureau balance data, credit balance data and previous application data –4 points (planned for the next two weeks)

♣ *Backlog2*: Engineer features to describe transaction and payment behavior –2 points (planned for the next two weeks)

♣ *Backlog3*: Visualize distributions of features generated –1 points (planned for the next two weeks)

♣ *Backlog4*: Identify the features having strong predictive power using xgboost embedded feature importance functionality –1 points

♣ *Backlog5*: Build at least 3 classification models including logistic regression, random forest and boosted tree to predict whether a customer will have late payment behavior –4 points

♣ *Backlog6*: Choose appropriate hyper-parameters considering both model complexity and model performance –4 points

♣ *Backlog7*: Compare the performance of all classification models with their best hyper-parameter combination using cross-validation –2 points

♣ *Icebox1*: Adjust classification threshold based on the risk preference of the bank. For instance, the bank can be risk-averse in one period and want to set threshold low such as 0.3 so that they will not lend to customers with predicted score higher than 0.3.

♣ *Icebox2*: Calculate the expected return if the bank grants a loan with the customer

o **Epic3**: Launch the customer profiling and prediction functionalities on a user-friendly website which can be used and maintained easily.

♣ *Backlog1*:  Write unit tests and have all tests passed locally

♣ *Backlog2*: Export dependencies of the model

♣ *Backlog3*: Move related data and file to AWS environment

♣ *Backlog4*: Write necessary backend structures using Flask

♣ *Backlog5*: Design frontend user interface

♣ *Backlog6*: Document every file clearly

## Repo structure 
```
├── README.md                         <- You are here
│
├── static                            <- CSS, JS files that remain static 
│
├── templates		                <- HTML (or other code) that is templated and changes based on a set of inputs
│
├── config                            <- Directory for yaml configuration files for model training, scoring, etc
│   ├── logging/                      <- Configuration files for python loggers
│
├── data                              <- Folder that contains data used or generated 
│
├── models                            <- Trained model objects (TMOs), model predictions, and/or model summaries
│   
├── notebooks
│   ├── develop                       <- Current notebooks being used in development
│   ├── deliver                       <- Notebooks shared with others
│   ├── archive                       <- Develop notebooks no longer being used
│
├── src                               <- Source code for the project 
│   ├── sql/                          <- SQL database and log file
│   ├── upload_data.py                <- Script for uploading the downloaded data to your own S3 bucket
│   ├── models.py                     <- Script for creating the sql database
│   ├── load_data.py                  <- Script for downloading data from public S3 to local and loading the data from local
│   ├── generate_features.py          <- Script for cleaning and transforming data and generating features used for use in training and scoring.
│   ├── train_model.py                <- Script for training machine learning model(s)
│   ├── score_model.py                <- Script for scoring new predictions using a trained model.
│   ├── evaluate_model.py             <- Script for evaluating model performance 
│   ├── test.py                       <- Script for unit testing of some functions
│   
├── test                              <- Files necessary for running unit tests 
├── app.py                            <- Flask wrapper for running the model 
├── config.py                         <- Configuration file for Flask app
├── requirements.txt                  <- Python package dependencies 
├── Makefile                          <- makefile for reproducing the project
```

## Running the application 
### 1. Set up environment 
The `requirements.txt` file contains the packages required to run the model code. An environment can be set up in two ways. 
#### With `virtualenv`
```bash
pip install virtualenv
virtualenv pennylane
source pennylane/bin/activate
pip install -r requirements.txt
```
#### With `conda`
```bash
conda create -n pennylane python=3.7
conda activate pennylane
pip install -r requirements.txt
```

### 2. Configure Flask app 
`config.py` holds the configurations for the Flask app. It includes the following configurations:
```python
DEBUG = True  # Keep True for debugging, change to False when moving to production 
LOGGING_CONFIG = "config/logging/local.conf"  # Path to file that configures Python logger
PORT = 3002  # What port to expose app on 
SQLALCHEMY_DATABASE_URI = 'sqlite:///src/sql/RiskPrediction.db'  # URI for database that contains tracks
```

### 3. Initialize the database 
To create the database in the location configured in `config.py`, cd to path_to_repo/src, run:
 ```bash
python models.py
 ```

### 4. Run the application
cd to path_to_repo first
if you want to run locally:
 ```bash
export SQLALCHEMY_DATABASE_URI='sqlite:///src/sql/RiskPrediction.db'
 ```
if you want to run on RDS:
 ```bash
export SQLALCHEMY_DATABASE_URI="{conn_type}://{user}:{password}@{host}:{port}/{DATABASE_NAME}"
 ```
then: 
 ```bash
python app.py 
 ```

### 5. Interact with the application 
Go to http://127.0.0.1:3002/ to interact with the current version of hte app. 


## Testing 
cd to path_to_repo, run `make test` from the command line in the main project repository. 
Tests exist in `src/test.py`


## Reproducing the project locally
cd to path_to_repo, create the virtual environment if you have not done that:
 ```bash
make pennylane-env/bin/activate    
source pennylane-env/bin/activate
 ```
then:
 ```bash
export SQLALCHEMY_DATABASE_URI='sqlite:///src/sql/RiskPrediction.db'
make all
 ```
