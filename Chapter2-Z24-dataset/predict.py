from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import json

# Rule: 1 to 8, or 0 to 7 are healthy cases
HEALTHY_STATES = np.arange(8)

# Active learning algorithms do not always output all labels, hence the need to compress them into an interval
# for example, if only classes 16 and 18 are queried in AL, then only labels 0 and 1 are expected by XGBoost (or any other classifier for that matter)
def map_numbers_to_interval(lst):
    sorted_unique_nums = sorted(set(lst))
    mapping = {}
    
    for index, num in enumerate(sorted_unique_nums):
        mapping[num] = index
    
    return mapping

def binarize(y):
    return np.array([0 if x in HEALTHY_STATES else 1 for x in y])

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
    # Map labels in y_train to 1 (damaged) or 0 (healthy)
    y_train = binarize(y_train)
    
    # Define parameter grids for each classifier
    param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30]}
    param_grid_xgb = {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 0.2]}
    param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    param_grid_lr = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2', 'elasticnet']}
    
    if method == 0:
        model = RandomForestClassifier()
        param_grid = param_grid_rf
    elif method == 1:
        model = xgb.XGBClassifier()
        param_grid = param_grid_xgb
    elif method == 2:
        model = SVC()
        param_grid = param_grid_svm
    elif method == 3:
        model = LogisticRegression()
        param_grid = param_grid_lr
    else:
        raise ValueError("Choose 0 for Random Forest, 1 for XGBoost, 2 for SVM, or 3 for Logistic Regression.")
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    
    # Save best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    with open(f'best_params_method_{method}.json', 'w') as f:
        json.dump({'best_params': best_params, 'best_score': best_score}, f)
    
    # Use best model to predict
    y_pred = grid_search.predict(x_test)

    return y_pred