import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def get_database():
    mongo_url = os.environ["MONGO_URL"]
    client = MongoClient(mongo_url)
    return client["Unipol"]
