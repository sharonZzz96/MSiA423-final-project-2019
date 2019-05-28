"""
Created on 5/10/19

@author: sharon

"""
import os
import sys
import logging
import pandas as pd

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as sql

import argparse

logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
logger = logging.getLogger('sql_db')


Base = declarative_base()

class RiskPrediction(Base):
    """ Defines the data model for the table `RiskPrediction. """

    __tablename__ = 'RiskPrediction'

    id = Column(Integer, primary_key=True, unique=True, nullable=False)
    days_birth = Column(Integer, unique=False, nullable=False)
    days_employed = Column(Integer, unique=False, nullable=False)
    days_employed_perc = Column(Float, unique=False, nullable=False)
    days_id_publish = Column(Integer, unique=False, nullable=False)
    days_last_phone_change =  Column(Integer, unique=False, nullable=False)
    buro_days_credit_mean = Column(Float, unique=False, nullable=False)
    buro_days_credit_enddate_mean =  Column(Float, unique=False, nullable=False)
    annuity_income_perc = Column(Float, unique=False, nullable=False)
    income_credit_perc = Column(Float, unique=False, nullable=False)
    payment_rate = Column(Float, unique=False, nullable=False)
    instal_days_entry_payment_mean = Column(Float, unique=False, nullable=False)
    instal_dbd_mean = Column(Float, unique=False, nullable=False)
    instal_amt_payment_mean =  Column(Float, unique=False, nullable=False)
    approved_days_decision_mean =  Column(Float, unique=False, nullable=False)
    prediction = Column(String(100), unique=False, nullable=False)
    

    def __repr__(self):
        risk_repr = "<RiskPrediction(client_id='%s', prediction='%s')>"
        return risk_repr % (self.id, self.prediction)



def get_engine_string(RDS = False):
    if RDS:
        conn_type = "mysql+pymysql"
        user = os.environ.get("MYSQL_USER")
        password = os.environ.get("MYSQL_PASSWORD")
        host = os.environ.get("MYSQL_HOST")
        port = os.environ.get("MYSQL_PORT")
        DATABASE_NAME = 'msia423'
        engine_string = "{}://{}:{}@{}:{}/{}". \
            format(conn_type, user, password, host, port, DATABASE_NAME)
        # print(engine_string)
        logging.debug("engine string: %s"%engine_string)
        return  engine_string
    else:
        return 'sqlite:///RiskPrediction.db' # relative path



def create_db(args,engine=None):
    """Creates a database with the data models inherited from `Base`.

    Args:
        engine (:py:class:`sqlalchemy.engine.Engine`, default None): SQLAlchemy connection engine.
            If None, `engine_string` must be provided.
        engine_string (`str`, default None): String defining SQLAlchemy connection URI in the form of
            `dialect+driver://username:password@host:port/database`. If None, `engine` must be provided.

    Returns:
        None
    """
    if engine is None:
        RDS = eval(args.RDS) # evaluate string to bool
        logger.info("RDS:%s"%RDS)
        engine = sql.create_engine(get_engine_string(RDS = RDS))

    Base.metadata.create_all(engine)
    logging.info("database created")

    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create defined tables in database")
    parser.add_argument("--RDS", default="False",help="True if want to create in RDS else None")
    args = parser.parse_args()
    
    engine = create_db(args)

    # create engine
    #engine = sql.create_engine(get_engine_string(RDS = False))
    

    # create a db session
    Session = sessionmaker(bind=engine)  
    session = Session()

    user1 = RiskPrediction(days_birth=9296, days_employed=1968, days_employed_perc=0.2117, 
        days_id_publish=1952, days_last_phone_change=887, buro_days_credit_mean=518, 
        buro_days_credit_enddate_mean=207, annuity_income_perc=0.253, income_credit_perc=0.179, 
        payment_rate=0.045, instal_days_entry_payment_mean=568, instal_dbd_mean=10.78, 
        instal_amt_payment_mean=29638, approved_days_decision_mean=722, prediction=0.078)
    session.add(user1)

    user2 = RiskPrediction(days_birth=12597, days_employed=1656, days_employed_perc=0.1315, 
        days_id_publish=4117, days_last_phone_change=0, buro_days_credit_mean=975, 
        buro_days_credit_enddate_mean=801, annuity_income_perc=0.270, income_credit_perc=0.190, 
        payment_rate=0.051, instal_days_entry_payment_mean=576, instal_dbd_mean=10.60, 
        instal_amt_payment_mean=7386, approved_days_decision_mean=656, prediction=0.162)
    session.add(user2)

    user3 = RiskPrediction(days_birth=17504, days_employed=2147, days_employed_perc=0.1227, 
        days_id_publish=1066, days_last_phone_change=604, buro_days_credit_mean=1290, 
        buro_days_credit_enddate_mean=1188, annuity_income_perc=0.144, income_credit_perc=0.257, 
        payment_rate=0.037, instal_days_entry_payment_mean=289, instal_dbd_mean=9.51, 
        instal_amt_payment_mean=29840, approved_days_decision_mean=515, prediction=0.059)
    session.add(user3)

    session.commit()

    logger.info("Data added")

    query = "SELECT * FROM RiskPrediction"
    df = pd.read_sql(query, con=engine)
    logger.info(df)
    session.close()

