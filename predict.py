import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def predict(x_train, y_train, x_test, method=0):
    """
    Predict labels for x_test based on method chosen.

    Parameters:
    x_train (N by D array): Training features.
    y_train (1D array of length N): Training labels.
    x_test (M by D array): Test features.
    method (int): Algorithm to be used for prediction. 
                 0 for Random Forest, 1 for XGBoost, 2 for SVM, 3 for Logistic Regression.

    Returns:
    1D array of length M: Predicted labels for x_test.
    """

    if method == 0:
        model = RandomForestClassifier()
    elif method == 1:
        model = xgb.XGBClassifier()
    elif method == 2:
        model = SVC()
    elif method == 3:
        model = LogisticRegression()
    else:
        raise ValueError("Choose 0 for Random Forest, 1 for XGBoost, 2 for SVM, or 3 for Logistic Regression.")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_pred