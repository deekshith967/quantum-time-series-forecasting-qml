import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fillxy(data):
    X = []
    Y = []

    window = 10

    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(data)

    for i in range(len(data_scaled) - window):

        window_data = data_scaled[i:i + window]

        # convert (10,5) -> (5,10)
        window_data = window_data.T

        X.append(window_data)

        # predict OPEN price of next day
        Y.append(data_scaled[i + window][0])

    return np.array(X), np.array(Y)


def load_and_split_csv(filename, args):

    df = pd.read_csv(filename)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    df = df.dropna()

    data = df.values

    split = int(len(data) * (1 - args.test_percentage))

    train = data[:split]
    test = data[split:]

    return train, test


def load_dataset(args):

    dataset_file = {
        "sp500": "datasets/combined_dataset.csv",
        "nifty": "datasets/NIFTY50_Cleaned_Data.csv",
        "wti": "datasets/WTI_Offshore_Cleaned_Data.csv"
    }

    file = dataset_file[args.dataset]

    train, test = load_and_split_csv(file, args)
    X, Y = fillxy(train)

    tX, tY = fillxy(test)

    flattened = X.reshape((X.shape[0], -1))

    return X, Y, tX, tY, flattened


# 🔹 THIS FUNCTION MUST BE OUTSIDE load_dataset
def split_features_labels(X, Y, val_ratio=0.2):

    split_index = int(len(X) * (1 - val_ratio))

    train_features = X[:split_index]
    val_features = X[split_index:]

    train_labels = Y[:split_index]
    val_labels = Y[split_index:]

    return train_features, val_features, train_labels, val_labels