"""
Basic cleaning module test
"""
import pandas as pd
import pytest
import src.basic_cleaning


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    df = src.basic_cleaning.__clean_dataset(df)
    return df


def test_null(data):
    """
    Data is assumed to have no null values
    """
    assert data.shape == data.dropna().shape


def test_question_mark(data):
    """
    Data is assumed to have no question marks value
    """
    assert '?' not in data.values


def test_removed_columns(data):
    """
    Data is assumed to have no question marks value
    """
    assert "fnlgt" not in data.columns
    assert "education-num" not in data.columns
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns
