import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['feature1', 'feature2']]
    y = data['target']
    return X, y
