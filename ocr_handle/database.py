import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

def get_db():
    mongo_uri = os.getenv('MONGO_URI')
    client = MongoClient(mongo_uri, server_api=ServerApi('1'))
    return client.get_database("Senpro8")

def update_user_data(file_id, ocr_result):
    db = get_db()
    db.User_Data.update_one(
        {'_id': file_id},
        {'$set': {'ocr_result': ocr_result}}
    )
