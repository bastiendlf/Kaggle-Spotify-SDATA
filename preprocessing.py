import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from sklearn.preprocessing import scale
import numpy as np


def preprocess_data(df, features_columns, label=None, z_score=False, standardize=False):
    print("------------------------------------------")
    print("            Preprocessing data            ")
    print("------------------------------------------")
    print("Get dataset")
    print("Shape of the data to process : " + str(df.shape))
    print("------------------------------------------")

    # Create inputs and labels
    # label
    if label is not None:
        print("Extract labels ...")
        df_labels = df['genre']
        print("Encode labels ...")
        le = LabelEncoder()
        df_labels = le.fit_transform(df_labels)

    # inputs
    print("Extract inputs ...")
    df = df[features_columns]
    # Remove outliers
    if z_score:
        print("Remove outliers with zscore ...")
        z_scores = zscore(df)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 4).all(axis=1)
        df = df[filtered_entries]
        if label is not None:
            df_labels = df_labels[filtered_entries]

    # Strandardize : center reduce
    if standardize:
        print("Center and reduce inputs ...")
        df = scale(df, axis=0, with_mean=True, with_std=True)
        df = pd.DataFrame(df, columns=features_columns)

    print("------------------------------------------")
    print("Data shape after preprocessing : " + str(df.shape))
    if label is not None:
        print("Labels shape : " + str(df_labels.shape))

    print("Return dataset(s) ...")
    print("Preprocessing finished")
    print("------------------------------------------")

    if label is not None:
        pd.DataFrame(df_labels, columns=["genre"])
        res = (df, df_labels)
    else:
        res = df

    return res


def preprocess_data_exo2(df, features_columns, df_keep, z_score=False, standardize=False):
    print("------------------------------------------")
    print("            Preprocessing data exo2           ")
    print("------------------------------------------")
    print("Get dataset")

    print("------------------------------------------")

    # inputs
    print("Extract inputs ...")
    df_features = df[features_columns]
    print("Shape of the data to process : " + str(df_features.shape))
    # Remove outliers
    if z_score:
        print("Remove outliers with zscore ...")
        z_scores = zscore(df_features)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 4).all(axis=1)

        # Keeping non-outliers elements
        df_features = df_features[filtered_entries]
        df_keep = df_keep[filtered_entries]

    # Strandardize : center reduce
    if standardize:
        print("Center and reduce inputs ...")
        features_matrix = scale(df_features, axis=0, with_mean=True, with_std=True)
        df_features = pd.DataFrame(features_matrix, columns=features_columns)

    print("------------------------------------------")
    print("Data shape after preprocessing : " + str(df_features.shape))

    print("Return dataset(s) ...")
    print("Preprocessing finished")
    print("------------------------------------------")

    return df_features, df_keep
