"""
Train model procedure
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import src.common_functions


def train_test_model():
    """
    Execute model training
    """
    df = pd.read_csv("data/prepared/census.csv")
    train, _ = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = src.common_functions.process_data(
        train, categorical_features=src.common_functions.get_cat_features(),
        label="salary", training=True
    )
    trained_model = src.common_functions.train_model(X_train, y_train)

    dump(trained_model, "data/model/model.joblib")
    dump(encoder, "data/model/encoder.joblib")
    dump(lb, "data/model/lb.joblib")
