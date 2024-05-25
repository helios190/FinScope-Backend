import os
import time
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv

load_dotenv()
endpoint = os.getenv('CV_ENDPOINT')
key = os.getenv('CV_KEY')
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

def ocr_image_from_blob(blob_client):
    blob_data = blob_client.download_blob().readall()
    image_stream = BytesIO(blob_data)

    read_response = computervision_client.read_in_stream(image_stream, raw=True)

    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    while True:
        result = computervision_client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    extracted_text = []
    if result.status == OperationStatusCodes.succeeded:
        for text_result in result.analyze_result.read_results:
            for line in text_result.lines:
                extracted_text.append(line.text)
    
    return " ".join(extracted_text).replace("\n", " ")
