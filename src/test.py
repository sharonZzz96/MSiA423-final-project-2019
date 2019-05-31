import pandas as pd
import numpy as np
import os
import sklearn
import generate_features 
import evaluate_model


def test_choose_features_all():
	"""Test if choose_features_all can choose all features used in modeling"""
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
	"""Test if get_target can get correct target variable name"""
	path = os.getcwd()
	df_input = pd.read_csv(path+'/test/data/test_data.csv')
	y = generate_features.get_target(df_input, 'TARGET')

	assert np.array_equal(y, df_input['TARGET'].values)


def test_evaluate_model():
	"""Test if evaluate_model can generate correct confusion matrix"""
	path = os.getcwd()
	df_input = pd.read_csv(path+'/test/data/test_actual_prediction.csv')
	prediction = pd.read_csv(path+'/test/data/test_prediction.csv')
	kwarg_dic = {'metrics':['auc','accuracy']}
	confusion_df = evaluate_model.evaluate_model(df_input, prediction, **kwarg_dic)
	
	y_true = df_input.iloc[:,0]
	y_pred = prediction['ypred_bin_test']
	confusion = sklearn.metrics.confusion_matrix(y_true, y_pred)
	confusion_df2 = pd.DataFrame(confusion,
        index=['Actual negative','Actual positive'],
        columns=['Predicted negative', 'Predicted positive'])
	print(confusion_df2)

	assert confusion_df.equals(confusion_df2)


