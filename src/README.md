load_my_data.py and upload_data.py need to be run in AWS with the AWS credential in advance. You can clone the repo to the AWS environment.

src/load_my_data.py: download data from public S3 bucket

file_key: data/features.csv; data/application_train.csv; data/bureau_balance.csv; data/bureau.csv; data/credit_card_balance.csv; data/installments_payments.csv; data/POS_CASH_balance.csv; data/previous_application.csv

bucket_name: nw-sharonzhang

output_file_path: output path for downloaded data

command to run example: 1) cd path_to_repo/src 2) python load_my_data.py --file_key data/application_train.csv --bucket_name nw-sharonzhang --output_file_path ../data/application_train.csv


src/upload_data.py: upload data to your own S3 bucket

input_file_path: local path for uploaded data

bucket_name: your S3 bucket name

access_key_id: aws access key id of the s3 bucket

secret_access_key: aws secret access key of the s3 bucket

output_file_path: output path for uploaded file on S3

command to run example: 1) cd path_to_repo/src 2) python load_my_data.py --file_key data/application_train.csv --bucket_name nw-sharonzhang --access_key_id <your_access_key> --secret_access_key<your_secret_access_key> --output_file_path ../data/application_train.csv


src/models.py: create database

--RDS if you want to create database in RDS else None.
Note: created database can be checked in 'src/logfile'

command to run example: 1) cd path_to_repo/src/sql 2) python models.py --RDS

