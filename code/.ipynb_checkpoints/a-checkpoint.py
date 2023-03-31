'/red/ruogu.fang/jbroce/ADNI_3D/testA_labels.csv'

import pandas as pd
import os

def get_train(directory):
    csv_path = os.path.join(directory, "trainA_labels.csv")
    df = pd.read_csv(csv_path)

    image_paths = [os.path.join(directory, "trainA", filename) for filename in df['id']]
    labels = df['label'].tolist()

    return image_paths, labels
    
def get_test(directory):
    csv_path = os.path.join(directory, "testA_labels.csv")
    df = pd.read_csv(csv_path)

    image_paths = [os.path.join(directory, "testA", filename) for filename in df['id']]
    labels = df['label'].tolist()

    return image_paths, labels
def get_data(directory):
    csv_path = os.path.join(directory, "labels.csv")
    df = pd.read_csv(csv_path)

    image_paths = [os.path.join(directory, "images", filename) for filename in df['id']]
    labels = df['label'].tolist()
    return image_paths, labels
