import datetime
import pandas as pd
import concurrent.futures
import time
from pymongo import MongoClient
from google.protobuf.json_format import MessageToJson
import json
import numpy as np
from scipy.optimize import least_squares

def mongodb_init():
    client = MongoClient('mongodb://root:rootpassword@localhost:27017/')
    db = client['toa_measurements']
    collection = db['toa_measurements']
    return collection

collection = mongodb_init()

speed_of_light = 299792458 # in meters per second
sampling_rate = 61.44e6 # in samples per second

# Fetch the toaVal values from MongoDB and store them in a list
toa_values = []
for doc in collection.find():
    toa_values.append(doc['paramMap'][1]['toa']['toaVal'])

# convert toa_values to numpy array
toa_values = np.array(toa_values)

# Calculate distances
distances = [val * speed_of_light / sampling_rate for val in toa_values]

print(distances)
