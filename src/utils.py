import json
from sklearn.datasets import load_digits

def load_config(path="config/config.json"):
    with open(path, 'r') as f:
        return json.load(f)

def load_data():
    digits = load_digits()
    return digits.data, digits.target
