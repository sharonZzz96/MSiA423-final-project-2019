# Example project repository

<!-- toc -->

- [Project Charter](#project-charter)
- [Backlog](#backlog)
- [Repo structure](#repo-structure)
- [Documentation](#documentation)
- [Running the application](#running-the-application)
  * [1. Set up environment](#1-set-up-environment)
    + [With `virtualenv` and `pip`](#with-virtualenv-and-pip)
    + [With `conda`](#with-conda)
  * [2. Configure Flask app](#2-configure-flask-app)
  * [3. Initialize the database](#3-initialize-the-database)
  * [4. Run the application](#4-run-the-application)
- [Testing](#testing)

<!-- tocstop -->

## Project Charter 

**Vision**: The bank struggles to target customers with insufficient or non-existent credit histories and has a hard time to decide whether to grant loans to those customers. This project will help the bank evaluate the repayment ability of those “underserved” customers, expand the loan servicing and generate more revenue.

**Mission**: Profiling of customers who are likely to default will be provided to help the bank filter out high-risk customers in the first stage. The bank will also be able to classify a new customer into the category of “with payment difficult” and “without payment difficult” to decide if they want to grant loans to the customer. The dataset used to train the classification model contains personal background, previous credits and loan application with other institutions of approximately 300,000 applicants. 

**Success criteria**: 1). AUC, Sensitivity and Specificity of test data will be applied to measure the performance of the model. Considering the available models in the market whose AUCs are between 0.65-0.73, we will set the minimum value of test AUC as 0.70 for success. 2). Whether the additional revenue from lending to those underserved customers reaches the finance goal of the bank (e.g. less than 2% default risk, generating more than 1 million revenue from those customers) will be used to evaluate the business outcome.

## Backlog
**Themes**: There are distinct types of unbanked customers. Some of them have repayment capability and can contribute to the reliable source of income to the bank, and therefore not providing loan servicing for all of them will be a loss to the bank. Through this project with the customer profiling and classification model, the bank will be able to identify high-quality customers with insufficient credit history. The bank can establish positive and safe loaning relationship with them, generate higher revenue and have more business opportunity. 

o **Epic1**: Explore various types of customers unserved by the bank from different aspects to identify what type of customers are likely to have late payment 

♣ *Backlog1*: Compare the customers with late payment and without late payment based on their individual background data (e.g. income, family, housing, education) –2 points (planned for the next two weeks)

♣ *Backlog2*: Analyze the credit behavior from Credit Bureau of late-payment population and non-late-payment population –4 points (planned for the next two weeks)

♣ *Backlog3*: Analyze the previous loan applications of the customers and study the correlation between previous loan application situation and late-payment behavior of future loans –4 points 

♣ *Backlog4*: Compare the customers with late payment and without late payment based on their social surrounding data –2 points


o **Epic2**: Based on their background, credit and transaction information in other financial institutions, classify customers into two groups: with payment difficulty and without payment difficulty 

♣ *Backlog1*: Understand, clean and merge different datasets –4 points (planned for the next two weeks)

♣ *Backlog2*: Engineer features to describe transaction and payment behavior –2 points (planned for the next two weeks)

♣ *Backlog3*: Analyze distributions of features generated –1 points 

♣ *Backlog4*: Identify the features having strong predictive power –1 points

♣ *Backlog5*: Build classification model to predict whether a customer will have late payment behavior –4 points

♣ *Backlog6*: Choose appropriate hyper-parameters considering both model complexity and model performance –4 points

♣ *Backlog7*: Adjust classification threshold based on the risk preference of the bank –2 points

♣ *Icebox1*: Calculate the expected return if the bank grants a loan with the customer

## Repo structure 

```
├── README.md                         <- You are here
│
├── app
│   ├── static/                       <- CSS, JS files that remain static 
│   ├── templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs
│   ├── models.py                     <- Creates the data model for the database connected to the Flask app 
│   ├── __init__.py                   <- Initializes the Flask app and database connection
│
├── config                            <- Directory for yaml configuration files for model training, scoring, etc
│   ├── logging/                      <- Configuration files for python loggers
│
├── data                              <- Folder that contains data used or generated. Only the external/ and sample/ subdirectories are tracked by git. 
│   ├── archive/                      <- Place to put archive data is no longer usabled. Not synced with git. 
│   ├── external/                     <- External data sources, will be synced with git
│   ├── sample/                       <- Sample data used for code development and testing, will be synced with git
│
├── docs                              <- A default Sphinx project; see sphinx-doc.org for details.
│
├── figures                           <- Generated graphics and figures to be used in reporting.
│
├── models                            <- Trained model objects (TMOs), model predictions, and/or model summaries
│   ├── archive                       <- No longer current models. This directory is included in the .gitignore and is not tracked by git
│
├── notebooks
│   ├── develop                       <- Current notebooks being used in development.
│   ├── deliver                       <- Notebooks shared with others. 
│   ├── archive                       <- Develop notebooks no longer being used.
│   ├── template.ipynb                <- Template notebook for analysis with useful imports and helper functions. 
│
├── src                               <- Source data for the project 
│   ├── archive/                      <- No longer current scripts.
│   ├── helpers/                      <- Helper scripts used in main src files 
│   ├── sql/                          <- SQL source code
│   ├── add_songs.py                  <- Script for creating a (temporary) MySQL database and adding songs to it 
│   ├── ingest_data.py                <- Script for ingesting data from different sources 
│   ├── generate_features.py          <- Script for cleaning and transforming data and generating features used for use in training and scoring.
│   ├── train_model.py                <- Script for training machine learning model(s)
│   ├── score_model.py                <- Script for scoring new predictions using a trained model.
│   ├── postprocess.py                <- Script for postprocessing predictions and model results
│   ├── evaluate_model.py             <- Script for evaluating model performance 
│
├── test                              <- Files necessary for running model tests (see documentation below) 

├── run.py                            <- Simplifies the execution of one or more of the src scripts 
├── app.py                            <- Flask wrapper for running the model 
├── config.py                         <- Configuration file for Flask app
├── requirements.txt                  <- Python package dependencies 
```
This project structure was partially influenced by the [Cookiecutter Data Science project](https://drivendata.github.io/cookiecutter-data-science/).

## Documentation
 
* Open up `docs/build/html/index.html` to see Sphinx documentation docs. 
* See `docs/README.md` for keeping docs up to date with additions to the repository.

## Running the application 
### 1. Set up environment 

The `requirements.txt` file contains the packages required to run the model code. An environment can be set up in two ways. See bottom of README for exploratory data analysis environment setup. 

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
SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/tracks.db'  # URI for database that contains tracks

```


### 3. Initialize the database 

To create the database in the location configured in `config.py` with one initial song, run: 

`python run.py create --artist=<ARTIST> --title=<TITLE> --album=<ALBUM>`

To add additional songs:

`python run.py ingest --artist=<ARTIST> --title=<TITLE> --album=<ALBUM>`


### 4. Run the application 
 
 ```bash
 python app.py 
 ```

### 5. Interact with the application 

Go to [http://127.0.0.1:3000/]( http://127.0.0.1:3000/) to interact with the current version of hte app. 

## Testing 

Run `pytest` from the command line in the main project repository. 


Tests exist in `test/test_helpers.py`
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTExNDIzMTI4LDE5NDk1NDg0NzZdfQ==
-->