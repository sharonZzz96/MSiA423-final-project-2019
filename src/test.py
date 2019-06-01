import pandas as pd
import numpy as np
import os
import sklearn
import sklearn.metrics
import generate_features 
import train_model
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


def test_generate_features():
	"""Test if generate_features can generate all the features required for modeling"""
	path = os.getcwd()
	df1 = pd.read_csv(path+'/test/data/test_application_train.csv')
	df2 = pd.read_csv(path+'/test/data/test_bureau.csv')
	df3 = pd.read_csv(path+'/test/data/test_previous_application.csv')
	df4 = pd.read_csv(path+'/test/data/test_installments_payments.csv')
	kwarg_dic = {'choose_features': {'features_to_use': [['SK_ID_CURR', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_LAST_PHONE_CHANGE'],['SK_ID_CURR', 'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE'],['SK_ID_CURR', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION'],['SK_ID_CURR', 'AMT_PAYMENT', 'DAYS_ENTRY_PAYMENT', 'DAYS_INSTALMENT']],'target': 'TARGET'}}
	df = generate_features.generate_features(df1,df2,df3,df4,**kwarg_dic)
	output_col = list(df.columns.values)

	required_col = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_EMPLOYED_PERC', 'BURO_DAYS_CREDIT_MEAN', 'DAYS_ID_PUBLISH', 'ANNUITY_INCOME_PERC', 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'INSTAL_DBD_MEAN', 'PAYMENT_RATE', 'INCOME_CREDIT_PERC', 'INSTAL_AMT_PAYMENT_MEAN', 'APPROVED_DAYS_DECISION_MEAN', 'DAYS_LAST_PHONE_CHANGE', 'BURO_DAYS_CREDIT_ENDDATE_MEAN']
	# evaluate if output columns contain all required columns for modeling
	assert set(output_col ) >= set(required_col)


def test_split_data():
	"""Test if split_data can split data as the proportion specified """
	path = os.getcwd()
	df_input = pd.read_csv(path+'/test/data/test_data.csv')
	X, y = train_model.split_data(df_input, df_input, train_size=0.9, test_size=0.1)
	ratio_X = X['train'].shape[0] / X['test'].shape[0]
	ratio_y = y['train'].shape[0] / y['test'].shape[0]
	assert ((ratio_X == 9) and (ratio_y == 9))


def test_evaluate_model():
	"""Test if evaluate_model can generate correct confusion matrix"""
	path = os.getcwd()
	df_input = pd.read_csv(path+'/test/data/test_actual_prediction.csv')
	prediction = pd.read_csv(path+'/test/data/test_prediction.csv')
	# use evaluate model function
	kwarg_dic = {'metrics':['auc','accuracy']}
	confusion_df = evaluate_model.evaluate_model(df_input, prediction, **kwarg_dic)
	
	# manually generate stats
	y_true = df_input.iloc[:,0]
	y_pred = prediction['ypred_bin_test']
	result = pd.concat([y_true, y_pred], axis=1, sort=False)
	NN = result.loc[(result.TARGET==0) & (result.ypred_bin_test==0)].count()[1]
	NP = result.loc[(result.TARGET==0) & (result.ypred_bin_test==1)].count()[1]
	PN = result.loc[(result.TARGET==1) & (result.ypred_bin_test==0)].count()[1]
	PP = result.loc[(result.TARGET==1) & (result.ypred_bin_test==1)].count()[1]
	d = {'Predicted negative': [NN, PN], 'Predicted positive': [NP, PP]}
	df = pd.DataFrame(data=d,index=['Actual negative','Actual positive'])

	assert confusion_df.equals(df)







