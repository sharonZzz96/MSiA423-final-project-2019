src/load_my_data.py	<- download data from public S3 bucket
- file_key: data/features.csv	 <- organized feature for modeling
	    data/application_train.csv	  <- raw data
	    data/bureau_balance.csv
	    data/bureau.csv
	    data/credit_card_balance.csv
	    data/installments_payments.csv
	    data/POS_CASH_balance.csv
	    data/previous_application.csv
- bucket_name: nw-sharonzhang


src/upload_data.py	<- upload data to your own S3 bucket

src/sql/models.py	<- create database 