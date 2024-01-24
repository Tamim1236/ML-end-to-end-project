import sys
import os
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
  
  def initiate_model_trainer(self, train_array, test_array):
    try:
      logging.info("Splitting the train and test input data")
      X_train, y_train, X_test, y_test = (
        train_array[:,:-1], # remove last column, keep everything else
        train_array[:, -1], # last column (target labels for all examples)
        test_array[:, :-1],
        test_array[:, -1]
      )

      models = {
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression(),
        "K-Neighbors Classifier": KNeighborsRegressor(),
        "XGBClassifier": XGBRegressor(),
        "CatBoosting Classifier": CatBoostRegressor(verbose=False),
        "AdaBoost Classifier": AdaBoostRegressor()
      }

      model_report : dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

      best_model_score = max(sorted(model_report.values()))
      best_model_name = list(model_report.keys())[ list(model_report.values()).index(best_model_score) ]
      best_model = models[best_model_name]

      # in case the score is too low (no good model to properly fit the data)
      if best_model_score < 0.6:
        raise CustomException("No best model found")
      
      logging.info("Found best model for train and test datasets")

      save_object( 
        file_path= self.model_trainer_config.trained_model_file_path,
        obj = best_model
      )

      predicted = best_model.predict(X_test)
      score = r2_score(y_test, predicted)
      return score
    
    except Exception as e:
      raise CustomException(e, sys)