# Imports the Google Cloud client library
from google.cloud import storage
import os

current_path = os.getcwd()

# Instantiates a client
storage_client = storage.Client()

# The name for the new bucket
bucket_name = 'music-tracks'

# Creates the new bucket
# bucket = storage_client.create_bucket(bucket_name)

def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(source_file_name)
  print('File {} uploaded to {}.'.format(
      source_file_name,
      destination_blob_name))

for folder in os.listdir('./fma_medium'):	
	if os.path.isdir(current_path + '/fma_medium/' + folder):		
		for file in os.listdir(current_path + '/fma_medium/' + folder):
			file_path = current_path + '/fma_medium/' + folder + '/' + file
			upload_blob(bucket_name, file_path, file)
