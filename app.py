from flask import render_template, request, redirect, url_for
import logging.config
from app import db, app
from flask import Flask
from src.models import RiskPrediction
from flask_sqlalchemy import SQLAlchemy
import pickle
import xgboost
import pandas as pd
import numpy as np
import traceback

# Initialize the Flask application
app = Flask(__name__)

# Configure flask app from config.py
app.config.from_object('config')

# Define LOGGING_CONFIG in config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger("safe-to-loan")
logger.debug('Test log')

# Initialize the database
db = SQLAlchemy(app)


@app.route('/')
def index():
    """Main view that lists songs in the database.
    Create view into index page that uses data queried from Track database and
    inserts it into the msiapp/templates/index.html template.
    Returns: rendered html template
    """

    return render_template('index.html')


@app.route('/add', methods=['GET','POST'])
def add_entry():
    """View that process a POST with new user input
    :return: redirect to index page
    """

    # get data
    DAYS_BIRTH = -1 * float(request.form['DAYS_BIRTH'])
    DAYS_EMPLOYED = -1 * float(request.form['DAYS_EMPLOYED'])
    DAYS_ID_PUBLISH = -1 * float(request.form['DAYS_ID_PUBLISH'])
    DAYS_LAST_PHONE_CHANGE = -1 * float(request.form['DAYS_LAST_PHONE_CHANGE'])
    annuity = float(request.form['annuity'])
    credit = float(request.form['credit'])
    income = float(request.form['income'])
    BURO_DAYS_CREDIT_MEAN = -1 * float(request.form['BURO_DAYS_CREDIT_MEAN'])
    BURO_DAYS_CREDIT_ENDDATE_MEAN = float(request.form['BURO_DAYS_CREDIT_ENDDATE_MEAN'])
    INSTAL_DAYS_ENTRY_PAYMENT_MEAN = -1 * float(request.form['INSTAL_DAYS_ENTRY_PAYMENT_MEAN'])
    INSTAL_DBD_MEAN = float(request.form['INSTAL_DBD_MEAN'])
    APPROVED_DAYS_DECISION_MEAN = -1* float(request.form['APPROVED_DAYS_DECISION_MEAN'])
    INSTAL_AMT_PAYMENT_MEAN = float(request.form['INSTAL_AMT_PAYMENT_MEAN'])
    risk_level = request.form['risk_level']

    # process some feature
    DAYS_EMPLOYED_PERC = DAYS_EMPLOYED / DAYS_BIRTH
    ANNUITY_INCOME_PERC = annuity / income
    PAYMENT_RATE = annuity / credit
    INCOME_CREDIT_PERC = income / credit

    path_to_tmo = 'models/risk-prediction.pkl'
    with open(path_to_tmo, "rb") as f:
        model = pickle.load(f)

    # create predict df
    X = pd.DataFrame(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 
        'DAYS_EMPLOYED_PERC', 'PAYMENT_RATE', 'INCOME_CREDIT_PERC', 
        'ANNUITY_INCOME_PERC', 'BURO_DAYS_CREDIT_MEAN', 'BURO_DAYS_CREDIT_ENDDATE_MEAN',
        'APPROVED_DAYS_DECISION_MEAN', 'INSTAL_DBD_MEAN', 'INSTAL_AMT_PAYMENT_MEAN',
        'INSTAL_DAYS_ENTRY_PAYMENT_MEAN'])
    X.loc[0] = [DAYS_BIRTH, DAYS_EMPLOYED, DAYS_ID_PUBLISH, DAYS_LAST_PHONE_CHANGE, 
        DAYS_EMPLOYED_PERC, PAYMENT_RATE, INCOME_CREDIT_PERC, 
        ANNUITY_INCOME_PERC, BURO_DAYS_CREDIT_MEAN, BURO_DAYS_CREDIT_ENDDATE_MEAN,
        APPROVED_DAYS_DECISION_MEAN, INSTAL_DBD_MEAN, INSTAL_AMT_PAYMENT_MEAN,
        INSTAL_DAYS_ENTRY_PAYMENT_MEAN]

    # risk level
    if risk_level == "risk averse":
        threshold = 0.06
    elif risk_level == "risk neutral":
        threshold = 0.08
    elif risk_level == "risk loving":
        threshold = 0.1

    # prediction
    pred_prob = float(model.predict_proba(X)[:,1])

    print(type(pred_prob))
    print(pred_prob)
    if pred_prob > threshold:
        result = 'The client has high late payment risk based on your risk preference: ' + risk_level
    else:
        result = 'It is safe to loan based on your risk preference: ' + risk_level

    try:
        user1 = RiskPrediction(days_birth=DAYS_BIRTH, 
            days_employed=DAYS_EMPLOYED, 
            days_employed_perc=DAYS_EMPLOYED_PERC,
            days_id_publish=DAYS_ID_PUBLISH,
            days_last_phone_change=DAYS_LAST_PHONE_CHANGE,
            buro_days_credit_mean=BURO_DAYS_CREDIT_MEAN,
            buro_days_credit_enddate_mean=BURO_DAYS_CREDIT_ENDDATE_MEAN,
            annuity_income_perc=ANNUITY_INCOME_PERC,
            income_credit_perc=INCOME_CREDIT_PERC,
            payment_rate=PAYMENT_RATE,
            instal_days_entry_payment_mean=INSTAL_DAYS_ENTRY_PAYMENT_MEAN,
            instal_dbd_mean=INSTAL_DBD_MEAN,
            instal_amt_payment_mean=INSTAL_AMT_PAYMENT_MEAN,
            approved_days_decision_mean=APPROVED_DAYS_DECISION_MEAN,
            prediction=pred_prob)
        db.session.add(user1)
        db.session.commit()
        logger.info("New client evaluation added: %s", result)
        return render_template('index.html', result=result)
    except:
        traceback.print_exc()
        logger.warning("Not able to display evaluation, error page returned")
        return render_template('error1.html')


@app.route('/view', methods=['GET','POST'])
def view_client():
    try: 
        users = db.session.query(RiskPrediction.id, RiskPrediction.annuity_income_perc, 
            RiskPrediction.payment_rate, RiskPrediction.prediction).order_by(RiskPrediction.prediction)     
        return render_template('view.html', users=users)

    except:
        traceback.print_exc()
        logger.warning("Not able to display clients, error page returned")
        return render_template('error2.html')



if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])
