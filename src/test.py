import pandas as pd
import numpy as np
import os
import generate_features 


def test_choose_features_all():
	# get data input
	path = os.getcwd()
	df_input = pd.read_csv(path+'/test/data/test_data.csv')
	features_to_use = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_EMPLOYED_PERC', 
                    'BURO_DAYS_CREDIT_MEAN', 'DAYS_ID_PUBLISH', 'ANNUITY_INCOME_PERC', 
                    'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'INSTAL_DBD_MEAN', 'PAYMENT_RATE', 
                    'INCOME_CREDIT_PERC', 'INSTAL_AMT_PAYMENT_MEAN', 'APPROVED_DAYS_DECISION_MEAN', 
                    'DAYS_LAST_PHONE_CHANGE', 'BURO_DAYS_CREDIT_ENDDATE_MEAN']
	df = generate_features.choose_features_all(df_input, features_to_use)
	output = list(df.columns.values)
	
	# order of column name does not matter
	assert set(output) == set(features_to_use)


def test_get_target():
	# get data input
	path = os.getcwd()
	df_input = pd.read_csv(path+'/test/data/test_data.csv')
	y = generate_features.get_target(df_input, 'TARGET')

	assert np.array_equal(y, df_input['TARGET'].values)


if __name__ == '__main__':
	test_choose_features()
	test_get_target()