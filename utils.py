import os
import json
from datetime import datetime

def ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(obj, path:str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)