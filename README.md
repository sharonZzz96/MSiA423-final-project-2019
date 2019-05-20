src/load_my_data.py: download data from public S3 bucket

- file_key: data/features.csv;
	    data/application_train.csv;
	    data/bureau_balance.csv;
	    data/bureau.csv;
	    data/credit_card_balance.csv;
	    data/installments_payments.csv;
	    data/POS_CASH_balance.csv;
	    data/previous_application.csv
	    
- bucket_name: nw-sharonzhang

- output_file_path: output path for downloaded data


src/upload_data.py: upload data to your own S3 bucket

- input_file_path: local path for uploaded data

- bucket_name: your S3 bucket name

- output_file_path: output path for uploaded file on S3


src/sql/models.py: create database 

- RDS True if you want to create database in RDS else None.

Note: created database can be checked in 'src/sql/logfile'
