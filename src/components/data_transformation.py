import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
import os


@dataclass
class DataTransformationConfig:
  preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()

  def get_data_transformer_object(self):
    try:
      numerical_features = ["writing_score", "reading_score"]
      categorical_features = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
      ]

      # handle missing values and then create pipeline 
      numerical_pipeline = Pipeline(
        steps = [
          ("imputer", SimpleImputer(strategy="median")),
          ("scaler", StandardScaler())
        ]
      )

      #logging.info("Numerical features standard scaling completed")
      logging.info(f"Numerical features: {numerical_features}")

      categorical_pipeline = Pipeline(
        steps = [
          ("imputer", SimpleImputer(strategy="most_frequent")),
          ("one_hot_encoder", OneHotEncoder()),
          ("scaler", StandardScaler())
        ]
      )

      #logging.info("Categorical features encoding completed")
      logging.info(f"Categorical features: {categorical_features}")
    
      # combine numerical and categorical pipelines
      preprocessor = ColumnTransformer(
        [
          ("numerical_pipeline", numerical_pipeline, numerical_features),
          ("categorical_pipeline", categorical_pipeline, categorical_features)
        ]
      )

      return preprocessor

    except Exception as e:
      raise CustomException(e, sys)