import os
from azure.storage.blob import BlobServiceClient
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
blob_storage_connection_string = os.getenv('BLOB_STRING')
blob_storage_container_name = os.getenv('STORAGE_CONTAINER')
blob_service_client = BlobServiceClient.from_connection_string(blob_storage_connection_string)
container_client = blob_service_client.get_container_client(blob_storage_container_name)

def upload_file_to_blob(file):
    filename = secure_filename(file.filename)
    blob_name = f"uploads/{filename}"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(file)
    file_url = blob_client.url
    return file_url, filename

def get_blob_client(file_name):
    return container_client.get_blob_client(f"uploads/{file_name}")
