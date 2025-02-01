import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """Save Python object using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """Evaluate multiple ML models and return their R² scores."""
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            try:
                para = param[model_name]

                # Skip GridSearchCV for CatBoost
                if model_name == "CatBoosting Regressor":
                    model.fit(X_train, y_train)  # Directly train CatBoost
                else:
                    gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
                    gs.fit(X_train, y_train)
                    model.set_params(**gs.best_params_)
                    model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_model_score

            except Exception as e:
                print(f"⚠️ Skipping {model_name} due to error: {str(e)}")
                continue  # Skip this model if GridSearchCV fails

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)    
