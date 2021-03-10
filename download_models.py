import boto3
from src.config.credentials import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
def download():
    s3 = boto3.resource(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
    for i in range(FOLDS):
        s3.meta.client.download_file('melenoma-classifier-models', f'model_fold_{i}.bin', f'model_s3/model_fold_{i}.bin')
