"""
This is the churn_library.py procedure.
Artifact produced will be in images, logs and models folders.
"""

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(dataset_path):
    """
    returns dataframe for the csv found at pth

    input:
            dataset_path: a path to the csv
    output:
            dataframe: pandas dataframe
    """
    dataframe = pd.read_csv(dataset_path)
    dataframe["Churn"] = dataframe.Attrition_Flag.apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return dataframe


def perform_eda(dataframe):
    """
    perform eda on df and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    """
    column_names = ["Churn", "Customer_Age", "Marital_Status", "Total_Trans", "Heatmap"]
    for column_name in column_names:
        plt.figure(figsize=(20, 10))
        if column_name == "Churn":
            dataframe.Churn.hist()
        elif column_name == "Customer_Age":
            dataframe.Customer_Age.hist()
        elif column_name == "Marital_Status":
            dataframe.Marital_Status.value_counts("normalize").plot(kind="bar")
        elif column_name == "Total_Trans":
            sns.displot(dataframe.Total_Trans_Ct)
        elif column_name == "Heatmap":
            sns.heatmap(dataframe.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.savefig("images/eda/%s.jpg" % column_name)
        plt.close()


def encoder_helper(dataframe, category_lst):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            dataframe: pandas dataframe with new columns for
    """
    for category_name in category_lst:
        category_lst = []
        category_groups = dataframe.groupby(category_name).mean()["Churn"]
        for val in dataframe[category_name]:
            category_lst.append(category_groups.loc[val])
        dataframe["%s_%s" % (category_name, "Churn")] = category_lst
    return dataframe


def perform_feature_engineering(dataframe):
    """
    input:
              dataframe: pandas dataframe

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    y_df = dataframe["Churn"]
    x_df = pd.DataFrame()
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"]
    x_df[keep_cols] = dataframe[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y_df, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(data):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    y_train = data[0]
    y_test = data[1]
    y_train_preds_lr = data[2]
    y_train_preds_rf = data[3]
    y_test_preds_lr = data[4]
    y_test_preds_rf = data[5]

    classification_reports_data = {
        "Random_Forest": (
            "Random Forest Train",
            y_test,
            y_test_preds_rf,
            "Random Forest Test",
            y_train,
            y_train_preds_rf),
        "Logistic_Regression": (
            "Logistic Regression Train",
            y_train,
            y_train_preds_lr,
            "Logistic Regression Test",
            y_test,
            y_test_preds_lr)}
    for title, classification_data in classification_reports_data.items():
        plt.rc("figure", figsize=(5, 5))
        plt.text(0.01, 1.25, str(classification_data[0]), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    classification_data[1], classification_data[2])), {
                "fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.6, str(classification_data[3]), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    classification_data[4], classification_data[5])), {
                "fontsize": 10}, fontproperties="monospace")
        plt.axis("off")
        plt.savefig("images/results/%s.jpg" % title)
        plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig("images/%s/Feature_Importance.jpg" % output_pth)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=1000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"]
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image([y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf])

    feature_importance_plot(cv_rfc, x_test, "results")

    joblib.dump(cv_rfc.best_estimator_, "models/rfc_model.pkl")
    joblib.dump(lrc, "models/logistic_model.pkl")


if __name__ == "__main__":
    data_df = import_data("data/bank_data.csv")
    perform_eda(data_df)
    encoded_data_df = encoder_helper(data_df,
                                     ["Gender",
                                      "Education_Level",
                                      "Marital_Status",
                                      "Income_Category",
                                      "Card_Category"])
    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(
        encoded_data_df)
    train_models(x_train_, x_test_, y_train_, y_test_)
