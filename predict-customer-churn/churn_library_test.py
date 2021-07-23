"""
Testing module that will check the churn_library.py procedure.
Pylint is automatically launched by main procedure so simply launching this script will be enough.
Artifact produced will be in images, logs and models folders.
"""

import os
import logging
import sys
import glob

import pytest
import joblib

import churn_library

os.environ['QT_QPA_PLATFORM']='offscreen'
logging.basicConfig(
    filename="logs/churn_library.log",
    level=logging.INFO,
    filemode='w',
    format="%(asctime)s: %(name)s - %(levelname)s - %(message)s",
    force=True)


@pytest.fixture(name='dataframe_raw')
def dataframe_raw_():
    """
    raw dataframe fixture - returns the raw dataframe from initial dataset file
    """
    try:
        internal_dataframe_raw = churn_library.import_data(
            "data/bank_data.csv")
        logging.info("Raw dataframe fixture creation: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Raw dataframe fixture creation: The file wasn't found")
        raise err
    return internal_dataframe_raw


@pytest.fixture(name='dataframe_encoded')
def dataframe_encoded_(dataframe_raw):
    """
    encoded dataframe fixture - returns the encoded dataframe on some specific column
    """
    try:
        dataframe_encoded = churn_library.encoder_helper(
            dataframe_raw, [
                "Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"])
        logging.info("Encoded dataframe fixture creation: SUCCESS")
    except KeyError as err:
        logging.error(
            "Encoded dataframe fixture creation: Not existent column to encode")
        raise err
    return dataframe_encoded


@pytest.fixture(name='feature_sequences')
def feature_sequences_(dataframe_encoded):
    """
    feature sequences fixtures - returns 4 series containing features sequences
    """
    try:
        x_train, x_test, y_train, y_test = churn_library.perform_feature_engineering(
            dataframe_encoded)

        logging.info("Feature sequence fixture creation: SUCCESS")
    except BaseException:
        logging.error(
            "Feature sequences fixture creation: Sequences length mismatch")
        raise
    return x_train, x_test, y_train, y_test


def test_import(dataframe_raw):
    """
    test import function - test initial dataset import for raw data
    """
    try:
        assert dataframe_raw.shape[0] > 0
        assert dataframe_raw.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(dataframe_raw):
    """
    test perform eda function - test creation of images related eda
    """
    churn_library.perform_eda(dataframe_raw)
    for image_name in [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans",
        "Heatmap"]:
        try:
            with open("images/eda/%s.jpg" % image_name, 'r'):
                logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing perform_eda: generated images missing")
            raise err


def test_encoder_helper(dataframe_encoded):
    """
    test encoder helper function - test dataset encoding
    """
    try:
        assert dataframe_encoded.shape[0] > 0
        assert dataframe_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have rows and columns")
        raise err
    try:
        for column in [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category"]:
            assert column in dataframe_encoded
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't have the right encoded columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")
    return dataframe_encoded


def test_perform_feature_engineering(feature_sequences):
    """
    test feature engineering - test feature engineering of the dataframe
    """
    try:
        x_train = feature_sequences[0]
        x_test = feature_sequences[1]
        y_train = feature_sequences[2]
        y_test = feature_sequences[3]
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing feature_engineering: Sequences length mismatch")
        raise err
    return feature_sequences


def test_train_models(feature_sequences):
    """
    test train_models - check result of training process
    """
    churn_library.train_models(
        feature_sequences[0],
        feature_sequences[1],
        feature_sequences[2],
        feature_sequences[3])
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing testing_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The files waeren't found")
        raise err
    for image_name in [
        "Logistic_Regression",
        "Random_Forest",
        "Feature_Importance"]:
        try:
            with open("images/results/%s.jpg" % image_name, 'r'):
                logging.info("Testing testing_models (report generation): SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing testing_models (report generation): generated images missing")
            raise err


if __name__ == "__main__":
    for directory in ["logs", "images/eda", "images/results", "./models"]:
        files = glob.glob("%s/*" % directory)
        for file in files:
            os.remove(file)
    sys.exit(pytest.main(["-s"]))
