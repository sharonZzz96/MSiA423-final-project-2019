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

command to run example: 1) cd path_to_repo/src  2) python load_my_data.py --file_key data/application_train.csv --bucket_name nw-sharonzhang --output_file_path <path_to_repo>/data/application_train.csv


src/upload_data.py	<- upload data to your own S3 bucket

command to run example: 1) cd path_to_repo/src  2) python upload_data.py --input_file_path <path_ti_repo>/data/application_train.csv --bucket_name <your S3 bucket name> --output_file_path data/application_train.csv

src/sql/models.py	<- create database 

command to run example: 1) cd path_to_repo/src/sql  2) python models.py --RDS True